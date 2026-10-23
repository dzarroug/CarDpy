"""Track acquired shot contributors in the same layout as saved diffusion images."""
import json
from pathlib import Path
import numpy as np
from cardpy.Data_Sorting import stacked2dictionary, dictionary2sorted, sorted2dictionary


class ShotMetadata:
    def __init__(self, shots, bvals, bvecs, slices, individual=False):
        if len(shots) != len(bvals):
            raise ValueError('Shot metadata must match input diffusion volumes')
        if any(not np.isfinite(float(s['trigger_ms'])) for s in shots):
            raise ValueError('Shot trigger times must be finite')
        self.shots = shots
        self.groups = {i: [i] for i in range(len(shots))}
        ids = np.broadcast_to(np.arange(len(shots)), (1, 1, slices, len(shots)))
        self.layout, _, _ = dictionary2sorted(stacked2dictionary(ids, bvals, bvecs),
                                              mode='individual' if individual else 'strict')
        self.decisions = {}

    def reject(self, keep, bvals, bvecs):
        if isinstance(keep, dict):
            entries = sorted2dictionary(self.layout, bvals, bvecs)
            for key, encoding in entries.items():
                self._decisions(encoding['images'][0, 0], keep[key])
        else:
            self._decisions(self.layout[0, 0], keep)

    def _decisions(self, ids, keep):
        if ids.shape != keep.shape:
            raise ValueError('Rejection decisions do not match shot provenance')
        for index in np.ndindex(ids.shape):
            shot = int(ids[index])
            accepted = bool(keep[index])
            row = keep[index[:-1]]
            fallback = not row.any() and index[-1] == 0
            for source in self.groups[shot]:
                self.decisions[(index[0], source)] = dict(
                    rejection_accepted=accepted, retained_as_fallback=fallback)

    def regroup(self, bvals, bvecs):
        return sorted2dictionary(self.layout, bvals, bvecs)

    def from_encodings(self, entries, individual=True):
        self.layout, _, _ = dictionary2sorted(entries, mode='individual' if individual else 'strict')

    def reorder(self, order):
        if order.shape != self.layout.shape[2:]:
            raise ValueError('Respiratory ordering does not match shot provenance')
        old = self.layout.copy()
        for slc, dif in np.ndindex(order.shape[:2]):
            # Unavailable padding has no contributor; original shots remain in the audit catalog.
            for rep, index in enumerate(order[slc, dif]):
                self.layout[0, 0, slc, dif, rep] = old[0, 0, slc, dif, index] if index >= 0 else -1

    def average(self, matrix, bvals, bvecs, individual=False):
        ids = self.regroup(bvals, bvecs) if individual else None
        data = sorted2dictionary(matrix, bvals, bvecs) if individual else None
        if individual:
            output = np.empty((1, 1, matrix.shape[2], len(ids), 1), int)
            for dif, (key, encoding) in enumerate(ids.items()):
                for slc in range(matrix.shape[2]):
                    valid = np.isfinite(data[key]['images'][:, :, slc]).any(axis=(0, 1))
                    output[0, 0, slc, dif, 0] = self._merge(encoding['images'][0, 0, slc, valid])
        else:
            output = np.empty((1, 1, matrix.shape[2], matrix.shape[3], 1), int)
            for slc, dif in np.ndindex(matrix.shape[2:4]):
                valid = np.isfinite(matrix[:, :, slc, dif]).any(axis=(0, 1))
                output[0, 0, slc, dif, 0] = self._merge(self.layout[0, 0, slc, dif, valid])
        self.layout = output

    def _merge(self, ids):
        sources = [s for i in ids if i >= 0 for s in self.groups[int(i)]]
        if not sources:
            return -1
        key = len(self.groups)
        self.groups[key] = sources
        return key

    def save(self, path, matrix, bvals, bvecs, header):
        if self.layout.shape[2:] != matrix.shape[2:]:
            raise ValueError('Saved diffusion layout does not match shot provenance')
        volumes, times = [], []
        for rep in range(matrix.shape[4]):
            for dif in range(matrix.shape[3]):
                for slc in range(matrix.shape[2]):
                    key = int(self.layout[0, 0, slc, dif, rep])
                    available = bool(np.isfinite(matrix[:, :, slc, dif, rep]).any())
                    sources = self.groups.get(key, [])
                    trigger = [float(self.shots[i]['trigger_ms']) for i in sources]
                    row = dict(volume_index=rep * matrix.shape[3] + dif,
                               slice_index=slc, encoding_index=dif, repetition_index=rep,
                               bval=float(bvals[dif]), bvec=np.asarray(bvecs[dif]).tolist(),
                               available=available, source_shot_indices=sources,
                               trigger_times_ms=trigger,
                               trigger_time_ms=float(np.median(trigger)) if trigger else None,
                               valid_voxels=int(np.isfinite(matrix[:, :, slc, dif, rep]).sum()))
                    volumes.append(row)
                    if available:
                        times.extend(trigger)
        if times:
            header['Trigger Time'] = float(np.median(times))
            header['Trigger Time Minimum'] = min(times)
            header['Trigger Time Maximum'] = max(times)
        else:
            for key in ('Trigger Time', 'Trigger Time Minimum', 'Trigger Time Maximum'):
                header.pop(key, None)
        header['Shot Metadata File'] = Path(path).name
        catalog = [dict(s, shot_index=i) for i, s in enumerate(self.shots)]
        decisions = [dict(slice_index=slc, shot_index=i, **decision)
                     for (slc, i), decision in sorted(self.decisions.items())]
        payload = dict(schema_version=1, volume_order='repetition-major; matches NIfTI/bvals/bvecs',
                       trigger_units='ms', shots=catalog, volumes=volumes,
                       trigger_summary_ms=dict(median=float(np.median(times)),
                                               minimum=min(times), maximum=max(times)) if times else None,
                       rejection_decisions=decisions,
                       acquisition=header.get('_Acquisition Metadata', {}))
        Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
