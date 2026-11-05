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
        features = []
        with torch.inference_mode():
            for batch in np.array_split(pairs, max(1, (len(pairs) + 31) // 32)):
                images = torch.from_numpy(spectra[batch]).float().cuda()
                images = F.interpolate(images, size=(64, 256), mode='bilinear', align_corners=False)
                images = ((images - mean) / std).repeat_interleave(3, dim=1)
                features.append(model.cnn(images).cpu())
        features = torch.cat(features).clone()
        assert features.shape == (len(pairs), 1024, 1, 4)
        assert torch.isfinite(features).all() and np.isfinite(labels).all()
        record = dict(name=name, segment=number, features=features,
                      labels=torch.from_numpy(labels.astype(np.float32)), pairs=pairs,
                      world_radar=world, deltas=delta, timestamps=timestamps)
        segments.append(record)
    print('PREPARED', name, 'segments', len(segments), 'pairs', sum(len(s['labels']) for s in segments), flush=True)
    return segments


def predict(model, segment):
    model.eval()
    with torch.inference_mode():
        return torch.cat([
            model.forward_features(batch.cuda()[:, None])[:, 0].cpu()
            for batch in segment['features'].split(64)
        ])


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
    model.load_cnn(ROOT / 'weights/cnn_torch.pt')
    model.eval()
    assert all(not parameter.requires_grad for parameter in model.cnn.parameters())
    assert all(parameter.requires_grad for parameter in model.rotation.parameters())
    cnn_hash = hashlib.sha256((ROOT / 'weights/cnn_torch.pt').read_bytes()).hexdigest()
    protocol = dict(input='single-chip ADC range-azimuth power spectrum; radar only',
                    network='existing PyTorch radar-only model with radar self-attention',
                    adaptations=['spectrum replaces pointcloud range image', 'IMU and cross-modal branches removed',
                                 'radar self-attention retained', 'previous spectrum normalization retained',
                                 'ColoRadar split and 0.2 second frame spacing'],
                    cnn='author pretrained CNN, frozen', cnn_sha256=cnn_hash,
                    seed=seed, epochs=args.epochs, batch_pairs=64, time_steps=1,
                    sampling='shuffle files/segments; shuffle candidate starts; contiguous 64-pair batches',
                    optimizer='RMSprop lr=1e-5 alpha=.9 eps=1e-8',
                    lr_schedule='multiply by .75 after every 25 global epochs',
                    loss='translation MSE + .001 * XYZ Euler angle MSE in degrees',
                    normalization=dict(mean=mean, std=std), split=split,
                    selection='lowest pooled validation weighted single-step MSE',
                    no_yaw_pretraining=True, no_turn_resampling=True,
                    scope='author-style training for the agreed pure-radar spectrum adaptation')
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
                features = segment['features'][start:start + 64].cuda()[:, None]
                target = segment['labels'][start:start + 64].cuda()
                optimizer.zero_grad(set_to_none=True)
                prediction = model.forward_features(features)[:, 0]
                value = loss(prediction, target)
                assert torch.isfinite(value)
                value.backward()
                optimizer.step()
                updates += 1
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
