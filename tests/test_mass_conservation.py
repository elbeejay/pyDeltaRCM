import sys
import os
import numpy as np
from pathlib import Path

# add the project root to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyDeltaRCM.model import DeltaModel
from tests import utilities


def run_mass_balance_check(tmp_path, timesteps=1, toggle_subsidence=False, verbose=False):
    """
    Runs a mass balance check on the model.
    """
    # create a delta model with a temporary yaml file
    config = {'toggle_subsidence': toggle_subsidence}
    p = utilities.yaml_from_dict(Path(tmp_path), 'input.yaml', config)
    delta = DeltaModel(input_file=p)

    # get initial bed elevation
    eta_initial = np.copy(delta.eta)
    total_subsidence_volume = 0

    # run the model for multiple timesteps
    for _ in range(timesteps):
        delta.update()
        if toggle_subsidence and (delta.time >= delta.start_subsidence):
            total_subsidence_volume += np.sum(delta.sigma) * (delta.dx ** 2)

    # get the final bed elevation
    eta_final = delta.eta

    # calculate the change in bed elevation
    delta_eta = eta_final - eta_initial

    # calculate the total volume of deposited sediment
    deposited_volume = np.sum(delta_eta) * (delta.dx ** 2)

    # get the input sediment volume for all timesteps
    input_volume = delta.dVs * timesteps

    # calculate the volume change at the inlet
    inlet_cell_indices = np.where(delta.cell_type == 1)
    inlet_volume_change = np.sum(delta_eta[inlet_cell_indices]) * (delta.dx ** 2)

    # calculate the mass balance discrepancy
    discrepancy = (deposited_volume - inlet_volume_change +
                   delta.lost_sediment_volume - input_volume + total_subsidence_volume)

    if verbose:
        print(f"Mass Balance Check for {timesteps} timesteps with subsidence {toggle_subsidence}")
        print("===================================")
        print(f"Deposited volume: {deposited_volume}")
        print(f"Inlet volume change: {inlet_volume_change}")
        print(f"Lost sediment volume: {delta.lost_sediment_volume}")
        print(f"Total subsidence volume: {total_subsidence_volume}")
        print(f"Input volume: {input_volume}")
        print(f"Discrepancy: {discrepancy}")

    return discrepancy


def run_mass_balance_check_from_checkpoint(timesteps=1, verbose=False):
    """
    Runs a mass balance check on the model from a checkpoint.
    """
    # create a delta model from a checkpoint
    delta = DeltaModel(resume_checkpoint='docs/source/_resources/checkpoint/')

    # get initial bed elevation
    eta_initial = np.copy(delta.eta)
    total_subsidence_volume = 0

    # run the model for multiple timesteps
    for _ in range(timesteps):
        delta.update()
        if delta.toggle_subsidence and (delta.time >= delta.start_subsidence):
            total_subsidence_volume += np.sum(delta.sigma) * (delta.dx ** 2)

    # get the final bed elevation
    eta_final = delta.eta

    # calculate the change in bed elevation
    delta_eta = eta_final - eta_initial

    # calculate the total volume of deposited sediment
    deposited_volume = np.sum(delta_eta) * (delta.dx ** 2)

    # get the input sediment volume for all timesteps
    input_volume = delta.dVs * timesteps

    # calculate the volume change at the inlet
    inlet_cell_indices = np.where(delta.cell_type == 1)
    inlet_volume_change = np.sum(delta_eta[inlet_cell_indices]) * (delta.dx ** 2)

    # calculate the mass balance discrepancy
    discrepancy = (deposited_volume - inlet_volume_change +
                   delta.lost_sediment_volume - input_volume + total_subsidence_volume)

    if verbose:
        print(f"Mass Balance Check from checkpoint for {timesteps} timesteps")
        print("==========================================================")
        print(f"Deposited volume: {deposited_volume}")
        print(f"Inlet volume change: {inlet_volume_change}")
        print(f"Lost sediment volume: {delta.lost_sediment_volume}")
        print(f"Total subsidence volume: {total_subsidence_volume}")
        print(f"Input volume: {input_volume}")
        print(f"Discrepancy: {discrepancy}")

    return discrepancy


if __name__ == '__main__':
    # create a temporary directory to run the model in
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_path:
        run_mass_balance_check(tmp_path, timesteps=10, verbose=True)
        run_mass_balance_check(tmp_path, timesteps=10, toggle_subsidence=True, verbose=True)
        run_mass_balance_check_from_checkpoint(timesteps=10, verbose=True)
