"""Freeze a root-chosen common fit cap after all train-only resource probes exist."""

import argparse
import copy
import json
from pathlib import Path
import sys

import multifield_protocol_guard as guard


def freeze(preprobe_path, preprobe_sha256, budget_seconds, root_authorized):
    guard.require(root_authorized is True, 'The caller must explicitly record root authorization')
    budget = guard.positive(budget_seconds, 'Root-chosen common training budget')
    path = Path(preprobe_path).resolve()
    guard.require(path.is_relative_to(guard.ROOT), 'Preprobe protocol must be inside the workspace')
    reference = {'path': path.relative_to(guard.ROOT).as_posix(), 'sha256': preprobe_sha256}
    preprobe = guard.read_reference(reference, 'Preprobe protocol')
    guard.require(preprobe.get('phase') == 'resource_probes', 'Expected the resource-probes protocol')
    proposed = copy.deepcopy(preprobe)
    proposed.update({'phase': 'fit', 'training_budget_seconds': budget, 'preprobe_protocol': reference})
    evidence = []
    for domain in guard.DOMAINS:
        for strategy in guard.STRATEGIES:
            row = {'domain': domain, 'strategy': strategy}
            for kind in ('functional', 'stream'):
                artifact = guard.artifact_path(domain, strategy, kind)
                guard.require(artifact.is_file(), f'Missing {kind} evidence: {artifact}')
                row[kind] = {'path': artifact.relative_to(guard.ROOT).as_posix(), 'sha256': guard.sha(artifact)}
            evidence.append(row)
    proposed['probe_evidence'] = evidence
    proposed['budget_decision'] = {
        'root_authorized': True,
        'authorization_record': 'The tool caller supplied --root-authorized and chose --budget-seconds explicitly.',
        'basis': 'train_only_resource_stream', 'test_or_validation_accuracy_used': False,
        'uniform_budget_seconds': budget, 'minimum_updates': 100, 'minimum_complete_case_epochs': 2,
        'quality_gate_scope': guard.DECISION_SCOPE,
        'selected_checkpoint_policy': 'Minimum validation checkpoint may be earlier; report its exposure separately from all executed training work.',
        'convergence_or_accuracy_pass_claimed': False,
    }
    readiness = guard.verify_fit_readiness(proposed)
    proposed['budget_decision'].update({
        'resource_projections': readiness['projections'],
        'all_projected_minimums_feasible': readiness['all_projected_minimums_feasible'],
        'projection_interpretation': 'Resource estimates only. The common cap is root-chosen, never selected automatically; actual executed-work gates remain mandatory.',
    })
    # Verify the final object too; extra decision data must not weaken the evidence gate.
    guard.verify_fit_readiness(proposed)
    target = guard.OUT / 'native-scalar-v3-fit-protocol.json'
    with target.open('x', encoding='utf8') as stream:
        stream.write(json.dumps(proposed, indent=2, allow_nan=False) + '\n')
    return {'status': 'fit_protocol_frozen_from_root_resource_decision',
            'path': target.relative_to(guard.ROOT).as_posix(), 'sha256': guard.sha(target),
            'training_budget_seconds': budget, 'all_projected_minimums_feasible': readiness['all_projected_minimums_feasible'],
            'artifact_hashes_verified': 12, 'convergence_or_accuracy_pass_claimed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preprobe-protocol', required=True)
    parser.add_argument('--preprobe-sha256', required=True)
    parser.add_argument('--budget-seconds', type=float, required=True)
    parser.add_argument('--root-authorized', action='store_true', required=True,
                        help='Record that root explicitly chose this resource-only common cap.')
    args = parser.parse_args()
    before = set(sys.modules)
    result = freeze(args.preprobe_protocol, args.preprobe_sha256, args.budget_seconds, args.root_authorized)
    guard.require(not any(name.split('.')[0] in {'numpy', 'torch', 'physicsnemo'} for name in set(sys.modules) - before),
                  'Decision generation imported an array/framework runtime')
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
