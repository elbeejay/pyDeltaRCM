import pytest
import os
import numpy as np

from .. import utilities
from pyDeltaRCM import DeltaModel
from pyDeltaRCM import preprocessor
from netCDF4 import Dataset


class TestConsistentOutputsBetweenMerges:

    def test_bed_after_one_update(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'Length', 10.0)
        utilities.write_parameter_to_file(f, 'Width', 10.0)
        utilities.write_parameter_to_file(f, 'seed', 0)
        utilities.write_parameter_to_file(f, 'dx', 1.0)
        utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
        utilities.write_parameter_to_file(f, 'S0', 0.0002)
        utilities.write_parameter_to_file(f, 'itermax', 1)
        utilities.write_parameter_to_file(f, 'Np_water', 10)
        utilities.write_parameter_to_file(f, 'u0', 1.0)
        utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
        utilities.write_parameter_to_file(f, 'h0', 1.0)
        utilities.write_parameter_to_file(f, 'H_SL', 0.0)
        utilities.write_parameter_to_file(f, 'SLR', 0.001)
        utilities.write_parameter_to_file(f, 'Np_sed', 10)
        utilities.write_parameter_to_file(f, 'f_bedload', 0.5)
        utilities.write_parameter_to_file(f, 'C0_percent', 0.1)
        utilities.write_parameter_to_file(f, 'toggle_subsidence', False)
        utilities.write_parameter_to_file(f, 'sigma_max', 0.0)
        utilities.write_parameter_to_file(f, 'start_subsidence', 50.)
        utilities.write_parameter_to_file(f, 'save_eta_figs', False)
        utilities.write_parameter_to_file(f, 'save_stage_figs', False)
        utilities.write_parameter_to_file(f, 'save_depth_figs', False)
        utilities.write_parameter_to_file(f, 'save_discharge_figs', False)
        utilities.write_parameter_to_file(f, 'save_velocity_figs', False)
        utilities.write_parameter_to_file(f, 'save_eta_grids', False)
        utilities.write_parameter_to_file(f, 'save_stage_grids', False)
        utilities.write_parameter_to_file(f, 'save_depth_grids', False)
        utilities.write_parameter_to_file(f, 'save_discharge_grids', False)
        utilities.write_parameter_to_file(f, 'save_velocity_grids', False)
        utilities.write_parameter_to_file(f, 'save_dt', 500)
        f.close()
        _delta = DeltaModel(input_file=p)

        _delta.update()

        # slice is: _delta.eta[:5, 4]
        _exp = np.array([-1., -0.9152762, -1.0004134, -1., -1.])
        assert np.all(_delta.eta[:5, 4] == pytest.approx(_exp))

    def test_bed_after_ten_updates(self, tmp_path):
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'Length', 10.0)
        utilities.write_parameter_to_file(f, 'Width', 10.0)
        utilities.write_parameter_to_file(f, 'seed', 0)
        utilities.write_parameter_to_file(f, 'dx', 1.0)
        utilities.write_parameter_to_file(f, 'L0_meters', 1.0)
        utilities.write_parameter_to_file(f, 'S0', 0.0002)
        utilities.write_parameter_to_file(f, 'itermax', 1)
        utilities.write_parameter_to_file(f, 'Np_water', 10)
        utilities.write_parameter_to_file(f, 'u0', 1.0)
        utilities.write_parameter_to_file(f, 'N0_meters', 2.0)
        utilities.write_parameter_to_file(f, 'h0', 1.0)
        utilities.write_parameter_to_file(f, 'H_SL', 0.0)
        utilities.write_parameter_to_file(f, 'SLR', 0.001)
        utilities.write_parameter_to_file(f, 'Np_sed', 10)
        utilities.write_parameter_to_file(f, 'f_bedload', 0.5)
        utilities.write_parameter_to_file(f, 'C0_percent', 0.1)
        utilities.write_parameter_to_file(f, 'toggle_subsidence', False)
        utilities.write_parameter_to_file(f, 'sigma_max', 0.0)
        utilities.write_parameter_to_file(f, 'start_subsidence', 50.)
        utilities.write_parameter_to_file(f, 'save_eta_figs', False)
        utilities.write_parameter_to_file(f, 'save_stage_figs', False)
        utilities.write_parameter_to_file(f, 'save_depth_figs', False)
        utilities.write_parameter_to_file(f, 'save_discharge_figs', False)
        utilities.write_parameter_to_file(f, 'save_velocity_figs', False)
        utilities.write_parameter_to_file(f, 'save_eta_grids', False)
        utilities.write_parameter_to_file(f, 'save_stage_grids', False)
        utilities.write_parameter_to_file(f, 'save_depth_grids', False)
        utilities.write_parameter_to_file(f, 'save_discharge_grids', False)
        utilities.write_parameter_to_file(f, 'save_velocity_grids', False)
        utilities.write_parameter_to_file(f, 'save_dt', 500)
        f.close()
        _delta = DeltaModel(input_file=p)

        for _ in range(0, 10):
            _delta.update()

        # slice is: test_DeltaModel.eta[:5, 4]
        _exp = np.array([1.7, 0.394864, -0.95006764,  -1., -1.])
        assert np.all(_delta.eta[:5, 4] == pytest.approx(_exp))

    def test_long_multi_validation(self, tmp_path):
        # IndexError on corner.

        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'seed', 42)
        utilities.write_parameter_to_file(f, 'Length', 600.)
        utilities.write_parameter_to_file(f, 'Width', 600.)
        utilities.write_parameter_to_file(f, 'dx', 5)
        utilities.write_parameter_to_file(f, 'Np_water', 10)
        utilities.write_parameter_to_file(f, 'Np_sed', 10)
        utilities.write_parameter_to_file(f, 'f_bedload', 0.05)
        f.close()
        delta = DeltaModel(input_file=p)

        for _ in range(0, 3):
            delta.update()

        # slice is: test_DeltaModel.eta[:5, 62]
        _exp1 = np.array([-4.976912, -4.979, -7.7932253, -4.9805, -2.7937498])
        assert np.all(delta.eta[:5, 62] == pytest.approx(_exp1))

        for _ in range(0, 10):
            delta.update()

        _exp2 = np.array([-4.9614887, -3.4891236, -12.195051,  -4.6706524, -2.7937498])
        assert np.all(delta.eta[:5, 62] == pytest.approx(_exp2))
        delta.finalize()


class TestModelIsReproducible:

    def test_same_result_two_models(self, tmp_path):
        """Test consistency of two models initialized from same yaml."""
        p1 = utilities.yaml_from_dict(tmp_path, 'input_1.yaml',
                                      {'out_dir': tmp_path / 'out_dir_1',
                                       'seed': 10, 'save_sandfrac_grids': True})
        p2 = utilities.yaml_from_dict(tmp_path, 'input_2.yaml',
                                      {'out_dir': tmp_path / 'out_dir_2',
                                       'seed': 10, 'save_sandfrac_grids': True})

        # create and update first model
        ModelA = DeltaModel(input_file=p1)
        ModelA.update()
        ModelA.output_netcdf.close()
        # create and update second model
        ModelB = DeltaModel(input_file=p2)
        ModelB.update()
        ModelB.output_netcdf.close()

        # fields should be the same
        assert ModelA.time == ModelB.time
        assert ModelA._time_iter == ModelB._time_iter
        assert ModelA._save_iter == ModelB._save_iter
        assert ModelA._save_time_since_data == ModelB._save_time_since_data
        assert np.all(ModelA.uw == ModelB.uw)
        assert np.all(ModelA.ux == ModelB.ux)
        assert np.all(ModelA.uy == ModelB.uy)
        assert np.all(ModelA.depth == ModelB.depth)
        assert np.all(ModelA.stage == ModelB.stage)
        assert np.all(ModelA.sand_frac == ModelB.sand_frac)
        assert np.all(ModelA.active_layer == ModelB.active_layer)

    def test_same_models_matrix(self, tmp_path):
        """Test models that have same parameters."""
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'Length', 1000.0)
        utilities.write_parameter_to_file(f, 'Width', 2000.0)
        utilities.write_parameter_to_file(f, 'verbose', 1)
        utilities.write_parameter_to_file(f, 'dx', 50.0)
        utilities.write_parameter_to_file(f, 'L0_meters', 150.0)
        utilities.write_parameter_to_file(f, 'N0_meters', 250.0)
        utilities.write_parameter_to_file(f, 'f_bedload', 0.2)
        utilities.write_parameter_to_file(f, 'save_eta_grids', True)
        utilities.write_matrix_to_file(f, ['seed'], [[1, 1]])
        f.close()

        # use preprocessor to run models
        pp = preprocessor.Preprocessor(input_file=p, timesteps=2)
        pp.run_jobs()

        # look at outputs
        ModelA = Dataset(os.path.join(str(pp._file_list[0])[:-12],
                         'pyDeltaRCM_output.nc'), 'r', format='NETCDF4')
        ModelB = Dataset(os.path.join(str(pp._file_list[1])[:-12],
                         'pyDeltaRCM_output.nc'), 'r', format='NETCDF4')

        # check attributes of the netCDFs
        # final eta grid has same LxW shape
        assert ModelA['eta'][-1, :, :].shape == ModelB['eta'][-1, :, :].shape
        assert ModelA.variables.keys() == ModelB.variables.keys()
        # check a few pieces of metadata
        assert ModelA['meta']['L0'][:] == ModelB['meta']['L0'][:]
        assert ModelA['meta'].variables.keys() == ModelB['meta'].variables.keys()
        # final eta grids should be the same
        assert np.all(ModelA['eta'][-1, :, :].data == ModelB['eta'][-1, :, :].data)
        # close netCDF output files
        ModelA.close()
        ModelB.close()

    def test_same_models_diff_save_dt(self, tmp_path):
        """Test models that have same parameters but different save_dt."""
        file_name = 'user_parameters.yaml'
        p, f = utilities.create_temporary_file(tmp_path, file_name)
        utilities.write_parameter_to_file(f, 'out_dir', tmp_path / 'out_dir')
        utilities.write_parameter_to_file(f, 'Length', 1000.0)
        utilities.write_parameter_to_file(f, 'Width', 2000.0)
        utilities.write_parameter_to_file(f, 'seed', 1)
        utilities.write_parameter_to_file(f, 'verbose', 1)
        utilities.write_parameter_to_file(f, 'dx', 50.0)
        utilities.write_parameter_to_file(f, 'L0_meters', 150.0)
        utilities.write_parameter_to_file(f, 'N0_meters', 250.0)
        utilities.write_parameter_to_file(f, 'f_bedload', 0.2)
        utilities.write_parameter_to_file(f, 'save_eta_grids', True)
        utilities.write_matrix_to_file(f, ['save_dt'], [[0, 50000]])
        f.close()

        # use preprocessor to run models
        pp = preprocessor.Preprocessor(input_file=p, timesteps=2)
        pp.run_jobs()

        # look at outputs
        ModelA = Dataset(os.path.join(str(pp._file_list[0])[:-12],
                         'pyDeltaRCM_output.nc'), 'r', format='NETCDF4')
        ModelB = Dataset(os.path.join(str(pp._file_list[1])[:-12],
                         'pyDeltaRCM_output.nc'), 'r', format='NETCDF4')

        # check attributes of the netCDFs
        # final eta grid has same LxW shape
        assert ModelA['eta'][-1, :, :].shape == ModelB['eta'][-1, :, :].shape
        assert ModelA.variables.keys() == ModelB.variables.keys()
        # check a few pieces of metadata
        assert ModelA['meta']['L0'][:] == ModelB['meta']['L0'][:]
        assert ModelA['meta'].variables.keys() == ModelB['meta'].variables.keys()
        # final eta grids should be the same
        assert np.all(ModelA['eta'][-1, :, :].data == ModelB['eta'][-1, :, :].data)
        # close netCDF output files
        ModelA.close()
        ModelB.close()
