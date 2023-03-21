import numpy as np

final_actual = np.load('test/predictions/binaries/binary_prediction_2021-10-12.npy', allow_pickle=True)
final_model_state = np.load('test/model/2018_2021_output/sim-003.npy', allow_pickle=True)

ones_actual = np.count_nonzero(final_actual)
zeroes_actual = final_actual.size - ones_actual

one_pc_actual = float(ones_actual) / final_actual.size * 100
zero_pc_actual = 100.0 - one_pc_actual

ones_simulated = np.count_nonzero(final_model_state)
zeroes_simulated = final_model_state.size - ones_simulated

one_pc_simulated = float(ones_simulated) / final_model_state.size * 100
zero_pc_simulated = 100.0 - one_pc_simulated

print()
print('Actual end:')
print('1: {:,d}  0: {:,d}'.format(ones_actual, zeroes_actual))
print('1: {:,.2f}%  0: {:,.2f}%'.format(one_pc_actual, zero_pc_actual))
print()
print('Simlulated end:')
print('1: {:,d}  0: {:,d}'.format(ones_simulated, zeroes_simulated))
print('1: {:,.2f}%  0: {:,.2f}%'.format(one_pc_simulated, zero_pc_simulated))
print()
