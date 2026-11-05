"""Author-style training on the agreed radar-only ColoRadar spectrum adaptation."""
import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp
from torch.nn import functional as F

from milliego.model import RadarOnlyMilliEgo
from scripts.validate_projection_diagnosis import metrics, save_tum

ROOT = Path('/home/metaiot_guest/Workspace/qinzx/milliEgo')
DATA = Path('/storage/data/metaiot_data/public_dataset/coloradar')
OLD = ROOT / 'runs/adc_heatmap_pilot_v1/coloradar_route_split_v1'


def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def atomic_checkpoint(path, value):
    temporary = path.with_suffix('.pt.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def loss(prediction, target):
    translation = (prediction[..., :3] - target[..., :3]).square().mean()
    rotation = (prediction[..., 3:] - target[..., 3:]).square().mean()
    return translation + 0.001 * rotation


def load_segments(name, output, model, mean, std):
    source = DATA / name
    cache = OLD / (name + '_spectra.npz')
    with np.load(cache) as saved:
        frames = saved['frames']
        spectra = saved['spectra']
    timestamps = np.loadtxt(source / 'single_chip/adc_samples/timestamps.txt')[frames]
    reference_times = np.loadtxt(source / 'groundtruth/timestamps.txt')
    reference_poses = np.loadtxt(source / 'groundtruth/groundtruth_poses.txt')
    extrinsic = np.load(ROOT / 'data/coloradar_lab0_native10_v1/labels.npz')['base_to_radar']
    world = np.repeat(np.eye(4)[None], len(frames), axis=0)
    world[:, :3, :3] = Slerp(reference_times, Rotation.from_quat(reference_poses[:, 3:7]))(timestamps).as_matrix()
    world[:, :3, 3] = np.column_stack([
        np.interp(timestamps, reference_times, reference_poses[:, axis]) for axis in range(3)
    ])
    world = world @ extrinsic
    breaks = np.flatnonzero((np.diff(frames) != 1) | (np.diff(timestamps) > 0.15)) + 1
    groups = np.split(np.arange(len(frames)), breaks)
    segments = []
    for number, group in enumerate(groups):
        chosen = group[::2]
        if len(chosen) < 2:
            continue
        # Downsample frames first, then pair adjacent frames, as in the author dataset.
        pairs = np.column_stack((chosen[:-1], chosen[1:]))
        delta = np.linalg.inv(world[pairs[:, 0]]) @ world[pairs[:, 1]]
        labels = np.column_stack((delta[:, :3, 3], Rotation.from_matrix(delta[:, :3, :3]).as_euler('XYZ', degrees=True)))
        record = dict(name=name, segment=number,
                      spectra=torch.from_numpy(spectra[chosen].astype(np.float32)).clone(),
                      labels=torch.from_numpy(labels.astype(np.float32)), pairs=pairs,
                      world_radar=world, deltas=delta, timestamps=timestamps, mean=mean, std=std)
        segments.append(record)
    print('PREPARED', name, 'segments', len(segments), 'pairs', sum(len(s['labels']) for s in segments), flush=True)
    return segments


def image_batch(segment, start, end):
    frames = segment['spectra'][start:end + 1].cuda()
    images = torch.stack((frames[:-1], frames[1:]), dim=1)
    images = F.interpolate(images, size=(64, 256), mode='bilinear', align_corners=False)
    return ((images - segment['mean']) / segment['std']).repeat_interleave(3, dim=1)[:, None]


def predict(model, segment):
    model.eval()
    with torch.inference_mode():
        return torch.cat([model(image_batch(segment, start, min(start + 64, len(segment['labels']))))[:, 0].cpu()
                          for start in range(0, len(segment['labels']), 64)])


def evaluate(model, segments, output=None):
    total, count = 0.0, 0
    records = []
    for segment in segments:
        prediction = predict(model, segment)
        value = float(loss(prediction, segment['labels']))
        total += value * len(prediction)
        count += len(prediction)
        if output is not None:
            score, trajectories = metrics(prediction.numpy(), segment, np.arange(len(prediction)), 'XYZ')
            records.append(dict(route=segment['name'], segment=segment['segment'], **score))
            for index, (estimated, reference, times) in enumerate(trajectories):
                stem = f"{segment['name']}_segment{segment['segment']}_{index}"
                save_tum(output / (stem + '_prediction.tum'), estimated, times)
                save_tum(output / (stem + '_reference.tum'), reference, times)
    return total / count, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=200)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    seed = 20261102
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(3)
    rng = np.random.default_rng(seed)
    split = json.loads((OLD / 'split.json').read_text())
    assert list(map(len, split.values())) == [40, 4, 8]
    assert len(set(sum(split.values(), []))) == 52
    # Use the same spectrum normalization as previous experiments to isolate training changes.
    old_checkpoint = torch.load(OLD / 'best.pt', weights_only=True, map_location='cpu')
    mean, std = float(old_checkpoint['mean']), float(old_checkpoint['std'])
    del old_checkpoint
    model = RadarOnlyMilliEgo().cuda()
    # Train every parameter from random initialization.
    model.eval()
    assert all(parameter.requires_grad for parameter in model.cnn.parameters())
    assert all(parameter.requires_grad for parameter in model.rotation.parameters())
    initial_cnn = model.cnn[0].conv.weight.detach().cpu().clone()
    protocol = dict(input='single-chip ADC range-azimuth power spectrum; radar only',
                    network='existing PyTorch radar-only model with radar self-attention',
                    adaptations=['spectrum replaces pointcloud range image', 'IMU and cross-modal branches removed',
                                 'radar self-attention retained', 'previous spectrum normalization retained',
                                 'ColoRadar split and 0.2 second frame spacing'],
                    cnn='random initialization; all parameters trainable; no pretrained weights loaded',
                    seed=seed, epochs=args.epochs, batch_pairs=64, time_steps=1,
                    sampling='shuffle files/segments; shuffle candidate starts; contiguous 64-pair batches',
                    optimizer='RMSprop lr=1e-5 alpha=.9 eps=1e-8',
                    lr_schedule='multiply by .75 after every 25 global epochs',
                    loss='translation MSE + .001 * XYZ Euler angle MSE in degrees',
                    normalization=dict(mean=mean, std=std), split=split,
                    selection='lowest pooled validation weighted single-step MSE',
                    no_yaw_pretraining=True, no_turn_resampling=True,
                    scope='author-style sampling/loss/schedule; trainable random CNN for radar-only spectra')
    atomic_json(output / 'protocol.json', protocol)
    train = sum([load_segments(name, output, model, mean, std) for name in split['train']], [])
    validation = sum([load_segments(name, output, model, mean, std) for name in split['validation']], [])
    eligible = [segment for segment in train if len(segment['labels']) >= 129]
    assert eligible and validation
    optimizer = torch.optim.RMSprop([p for p in model.parameters() if p.requires_grad], lr=1e-5, alpha=.9, eps=1e-8)
    history, best, updates = [], float('inf'), 0
    initial, _ = evaluate(model, validation)
    atomic_json(output / 'progress.json', dict(status='training', epoch=0, updates=0, initial_validation_loss=initial))
    print('TRAIN START', len(eligible), 'segments', sum(len(s['labels']) for s in eligible), 'pairs', flush=True)
    started = time.monotonic()
    for epoch in range(args.epochs):
        model.train()
        rate = 1e-5 * 0.75 ** (epoch // 25)
        for group in optimizer.param_groups:
            group['lr'] = rate
        total, count = 0.0, 0
        for index in rng.permutation(len(eligible)):
            segment = eligible[index]
            # Match upstream train_cross_att.py with batch_size=65 and N_frames=N_pairs+1.
            starts = np.arange(len(segment['labels']) - 65)
            rng.shuffle(starts)
            for batch_number in range(len(starts) // 64):
                start = int(starts[batch_number * 64])
                images = image_batch(segment, start, start + 64)
                target = segment['labels'][start:start + 64].cuda()
                optimizer.zero_grad(set_to_none=True)
                prediction = model(images)[:, 0]
                value = loss(prediction, target)
                assert torch.isfinite(value)
                value.backward()
                optimizer.step()
                updates += 1
                if updates == 1:
                    change = float((model.cnn[0].conv.weight.detach().cpu() - initial_cnn).abs().max())
                    assert change > 0, 'CNN did not update'
                    atomic_json(output / 'cnn_update_check.json', dict(first_step_max_parameter_change=change))
                    print('CNN_UPDATE_VERIFIED', change, flush=True)
                if updates % 100 == 0:
                    atomic_json(output / 'progress.json', dict(status='training', epoch=epoch + 1, updates=updates,
                                seconds=time.monotonic() - started))
                    print('STEP', updates, 'seconds', time.monotonic() - started, flush=True)
                total += float(value.detach()) * len(target)
                count += len(target)
        assert count > 0
        validation_loss, _ = evaluate(model, validation)
        record = dict(epoch=epoch + 1, updates=updates, lr=rate, train_loss=total / count,
                      validation_loss=validation_loss, seconds=time.monotonic() - started)
        history.append(record)
        payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch + 1,
                       updates=updates, mean=mean, std=std, rng=rng.bit_generator.state,
                       torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(), protocol=protocol)
        if validation_loss < best:
            best = validation_loss
            atomic_checkpoint(output / 'best.pt', payload)
        if (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            atomic_checkpoint(output / 'latest.pt', payload)
        atomic_json(output / 'history.json', history)
        atomic_json(output / 'progress.json', dict(status='training', best_validation_loss=best, **record))
        print('EPOCH', json.dumps(record), flush=True)
    checkpoint = torch.load(output / 'best.pt', weights_only=False, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    atomic_json(output / 'progress.json', dict(status='evaluating', selected_epoch=checkpoint['epoch']))
    results = []
    for name in split['test']:
        segments = load_segments(name, output, model, mean, std)
        _, records = evaluate(model, segments, output)
        results.extend(records)
        atomic_json(output / 'results.json', results)
        print('TEST', name, flush=True)
    atomic_json(output / 'progress.json', dict(status='complete', epochs=args.epochs,
                                              updates=updates, selected_epoch=checkpoint['epoch']))
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
