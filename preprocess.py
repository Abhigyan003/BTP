import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
import re
from shutil import copyfile

output_folder = 'processed'
data_folder = 'datasets'

datasets = ['synthetic', 'SMD', 'SWaT', 'SMAP', 'MSL', 'WADI', 'MSDS', 'UCR', 'MBA', 'NAB', 'CTF']

wadi_drop = ['2_LS_001_AL', '2_LS_002_AL','2_P_001_STATUS','2_P_002_STATUS']

def load_and_save(category, filename, dataset, dataset_folder):
    temp = np.genfromtxt(os.path.join(dataset_folder, category, filename),
                         dtype=np.float64,
                         delimiter=',')
    print(dataset, category, filename, temp.shape)
    np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)
    return temp.shape

def load_and_save2(category, filename, dataset, dataset_folder, shape):
	temp = np.zeros(shape)
	with open(os.path.join(dataset_folder, 'interpretation_label', filename), "r") as f:
		ls = f.readlines()
	for line in ls:
		pos, values = line.split(':')[0], line.split(':')[1].split(',')
		start, end, indx = int(pos.split('-')[0]), int(pos.split('-')[1]), [int(i)-1 for i in values]
		temp[start-1:end-1, indx] = 1
	print(dataset, category, filename, temp.shape)
	np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)

def normalize(a):
	a = a / np.maximum(np.absolute(a.max(axis=0)), np.absolute(a.min(axis=0)))
	return (a / 2 + 0.5)

def normalize2(a, min_a = None, max_a = None):
	if min_a is None: min_a, max_a = min(a), max(a)
	return (a - min_a) / (max_a - min_a), min_a, max_a

def normalize3(a, min_a = None, max_a = None):
	if min_a is None: min_a, max_a = np.min(a, axis = 0), np.max(a, axis = 0)
	return (a - min_a) / (max_a - min_a + 0.0001), min_a, max_a

def convertNumpy(df):
	x = df[df.columns[3:]].values[::10, :]
	return (x - x.min(0)) / (x.ptp(0) + 1e-4)

def load_data(dataset):
	folder = os.path.join(output_folder, dataset)
	os.makedirs(folder, exist_ok=True)
	dataset_folder = os.path.join(data_folder, dataset)
	if dataset == 'synthetic':
		train_file = os.path.join(data_folder, dataset, 'synthetic_data_with_anomaly-s-1.csv')
		test_labels = os.path.join(data_folder, dataset, 'test_anomaly.csv')
		dat = pd.read_csv(train_file, header=None)
		split = 10000
		train = normalize(dat.values[:, :split].reshape(split, -1))
		test = normalize(dat.values[:, split:].reshape(split, -1))
		lab = pd.read_csv(test_labels, header=None)
		lab[0] -= split
		labels = np.zeros(test.shape)
		for i in range(lab.shape[0]):
			point = lab.values[i][0]
			labels[point-30:point+30, lab.values[i][1:]] = 1
		test += labels * np.random.normal(0.75, 0.1, test.shape)
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{file}.npy'), eval(file))
	elif dataset == 'SMD':
		file_list = os.listdir(os.path.join(dataset_folder, "train"))
		for filename in file_list:
			if filename.endswith('.txt'):
				load_and_save('train', filename, filename.strip('.txt'), dataset_folder)
				s = load_and_save('test', filename, filename.strip('.txt'), dataset_folder)
				load_and_save2('labels', filename, filename.strip('.txt'), dataset_folder, s)
	elif dataset == 'UCR':
		file_list = os.listdir(dataset_folder)
		for filename in file_list:
			if not filename.endswith('.txt'): continue
			vals = filename.split('.')[0].split('_')
			dnum, vals = int(vals[0]), vals[-3:]
			vals = [int(i) for i in vals]
			temp = np.genfromtxt(os.path.join(dataset_folder, filename),
								dtype=np.float64,
								delimiter=',')
			min_temp, max_temp = np.min(temp), np.max(temp)
			temp = (temp - min_temp) / (max_temp - min_temp)
			train, test = temp[:vals[0]], temp[vals[0]:]
			labels = np.zeros_like(test)
			labels[vals[1]-vals[0]:vals[2]-vals[0]] = 1
			train, test, labels = train.reshape(-1, 1), test.reshape(-1, 1), labels.reshape(-1, 1)
			for file in ['train', 'test', 'labels']:
				np.save(os.path.join(folder, f'{dnum}_{file}.npy'), eval(file))
	elif dataset == 'NAB':
		file_list = os.listdir(dataset_folder)
		with open(dataset_folder + '/labels.json') as f:
			labeldict = json.load(f)
		for filename in file_list:
			if not filename.endswith('.csv'): continue
			df = pd.read_csv(dataset_folder+'/'+filename)
			vals = df.values[:,1]
			labels = np.zeros_like(vals, dtype=np.float64)
			for timestamp in labeldict['realKnownCause/'+filename]:
				tstamp = timestamp.replace('.000000', '')
				index = np.where(((df['timestamp'] == tstamp).values + 0) == 1)[0][0]
				labels[index-4:index+4] = 1
			min_temp, max_temp = np.min(vals), np.max(vals)
			vals = (vals - min_temp) / (max_temp - min_temp)
			train, test = vals.astype(float), vals.astype(float)
			train, test, labels = train.reshape(-1, 1), test.reshape(-1, 1), labels.reshape(-1, 1)
			fn = filename.replace('.csv', '')
			for file in ['train', 'test', 'labels']:
				np.save(os.path.join(folder, f'{fn}_{file}.npy'), eval(file))
	elif dataset == 'MSDS':
		df_train = pd.read_csv(os.path.join(dataset_folder, 'train.csv'))
		df_test  = pd.read_csv(os.path.join(dataset_folder, 'test.csv'))
		df_train, df_test = df_train.values[::5, 1:], df_test.values[::5, 1:]
		_, min_a, max_a = normalize3(np.concatenate((df_train, df_test), axis=0))
		train, _, _ = normalize3(df_train, min_a, max_a)
		test, _, _ = normalize3(df_test, min_a, max_a)
		labels = pd.read_csv(os.path.join(dataset_folder, 'labels.csv'))
		labels = labels.values[::1, 1:]
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{file}.npy'), eval(file).astype('float64'))
	elif dataset == 'SWaT':
		file = os.path.join(dataset_folder, 'series.json')
		df_train = pd.read_json(file, lines=True)[['val']][3000:6000]
		df_test  = pd.read_json(file, lines=True)[['val']][7000:12000]
		train, min_a, max_a = normalize2(df_train.values)
		test, _, _ = normalize2(df_test.values, min_a, max_a)
		labels = pd.read_json(file, lines=True)[['noti']][7000:12000] + 0
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{file}.npy'), eval(file))
	elif dataset in ['SMAP', 'MSL']:
		dataset_folder = os.path.join(data_folder, 'SMAP_MSL')
		file = os.path.join(dataset_folder, 'labeled_anomalies.csv')
		values = pd.read_csv(file)
		values = values[values['spacecraft'] == dataset]
		filenames = values['chan_id'].values.tolist()
		for fn in filenames:
			train = np.load(f'{dataset_folder}/train/{fn}.npy')
			test = np.load(f'{dataset_folder}/test/{fn}.npy')
			train, min_a, max_a = normalize3(train)
			test, _, _ = normalize3(test, min_a, max_a)
			np.save(f'{folder}/{fn}_train.npy', train)
			np.save(f'{folder}/{fn}_test.npy', test)
			labels = np.zeros(test.shape)
			indices = values[values['chan_id'] == fn]['anomaly_sequences'].values[0]
			indices = indices.replace(']', '').replace('[', '').split(', ')
			indices = [int(i) for i in indices]
			for i in range(0, len(indices), 2):
				labels[indices[i]:indices[i+1], :] = 1
			np.save(f'{folder}/{fn}_labels.npy', labels)
	elif dataset == 'WADI':
		ls = pd.read_csv(os.path.join(dataset_folder, 'WADI_attacklabels.csv'))
		train = pd.read_csv(os.path.join(dataset_folder, 'WADI_14days.csv'), skiprows=1000, nrows=2e5)
		test = pd.read_csv(os.path.join(dataset_folder, 'WADI_attackdata.csv'))
		train.dropna(how='all', inplace=True); test.dropna(how='all', inplace=True)
		train.fillna(0, inplace=True); test.fillna(0, inplace=True)
		test['Time'] = test['Time'].astype(str)
		test['Time'] = pd.to_datetime(test['Date'] + ' ' + test['Time'])
		labels = test.copy(deep = True)
		for i in test.columns.tolist()[3:]: labels[i] = 0
		for i in ['Start Time', 'End Time']: 
			ls[i] = ls[i].astype(str)
			ls[i] = pd.to_datetime(ls['Date'] + ' ' + ls[i])
		for index, row in ls.iterrows():
			to_match = row['Affected'].split(', ')
			matched = []
			for i in test.columns.tolist()[3:]:
				for tm in to_match:
					if tm in i: 
						matched.append(i); break			
			st, et = str(row['Start Time']), str(row['End Time'])
			labels.loc[(labels['Time'] >= st) & (labels['Time'] <= et), matched] = 1
		train, test, labels = convertNumpy(train), convertNumpy(test), convertNumpy(labels)
		print(train.shape, test.shape, labels.shape)
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{file}.npy'), eval(file))
	elif dataset == 'MBA':
		ls = pd.read_excel(os.path.join(dataset_folder, 'labels.xlsx'))
		train = pd.read_excel(os.path.join(dataset_folder, 'train.xlsx'))
		test = pd.read_excel(os.path.join(dataset_folder, 'test.xlsx'))
		train, test = train.values[1:,1:].astype(float), test.values[1:,1:].astype(float)
		train, min_a, max_a = normalize3(train)
		test, _, _ = normalize3(test, min_a, max_a)
		ls = ls.values[:,1].astype(int)
		labels = np.zeros_like(test)
		for i in range(-20, 20):
			labels[ls + i, :] = 1
		for file in ['train', 'test', 'labels']:
			np.save(os.path.join(folder, f'{file}.npy'), eval(file))
	elif dataset == 'CTF':
		# CTF dataset: 533 machines, 49 KPIs, 13 days total
		# First 5 days for training, next 8 days for testing
		# Data collected every 30 seconds, 2880 time points per day
		# Files are in CTF_data subdirectory with .txt extension
		# Named as: {machine_id}_{day}.txt (e.g., 463_18.txt)
		# Labels are in label_result/ as {machine_id}.pkl
		
		data_dir = os.path.join(dataset_folder, 'CTF_data')
		label_dir = os.path.join(dataset_folder, 'label_result')
		
		if not os.path.exists(data_dir):
			print(f"Error: {data_dir} not found. Please extract CTF_data.tar.gz first.")
			return
		
		data_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.txt') and not f.startswith('.')])
		
		# Group files by machine ID
		# Files are named: machine_day.txt (e.g., 463_18.txt, 463_19.txt, ...)
		machines = {}
		for fname in data_files:
			# Extract machine ID (everything before the last underscore)
			parts = fname.replace('.txt', '').rsplit('_', 1)
			if len(parts) == 2:
				machine_id = parts[0]
				if machine_id not in machines:
					machines[machine_id] = []
				machines[machine_id].append(fname)
		
		print(f"Found {len(machines)} machines in CTF dataset")
		
		# Process each machine
		for machine_id, machine_files in machines.items():
			# Sort files by day number
			machine_files.sort(key=lambda x: int(x.replace('.txt', '').rsplit('_', 1)[1]))
			
			if len(machine_files) != 13:
				print(f"Warning: Machine {machine_id} has {len(machine_files)} days instead of 13, skipping")
				continue
			
			# Load all days for this machine
			all_data = []
			for fname in machine_files:
				data = np.genfromtxt(os.path.join(data_dir, fname),
									dtype=np.float64,
									delimiter=',')
				# Handle NaN values by replacing with 0
				data = np.nan_to_num(data, nan=0.0)
				if data.ndim == 1:
					data = data.reshape(-1, 1)
				all_data.append(data)
			
			# Combine all days
			full_data = np.vstack(all_data)
			
			# Split: first 5 days (5 * 2880 = 14400 points) for train
			#        next 8 days (8 * 2880 = 23040 points) for test
			train_end = 5 * 2880
			test_end = train_end + (8 * 2880)
			
			train = full_data[:train_end]
			test = full_data[train_end:test_end]
			
			# Normalize
			train, min_a, max_a = normalize3(train)
			test, _, _ = normalize3(test, min_a, max_a)
			
			# Save train and test data
			np.save(f'{folder}/{machine_id}_train.npy', train)
			np.save(f'{folder}/{machine_id}_test.npy', test)
			
			# Load labels from label_result directory (pickle files)
			# Labels are point-level (1D), indicating anomaly at each time point
			label_file = os.path.join(label_dir, f'{machine_id}.pkl')
			
			if os.path.exists(label_file):
				with open(label_file, 'rb') as f:
					labels = pickle.load(f)
				labels = np.array(labels, dtype=np.float64).flatten()
				
				# Labels might be 23039 instead of 23040 (off-by-one in dataset)
				# Pad to match test length if needed
				if len(labels) < test.shape[0]:
					labels = np.pad(labels, (0, test.shape[0] - len(labels)), mode='constant', constant_values=0)
				elif len(labels) > test.shape[0]:
					labels = labels[:test.shape[0]]
				
				# Broadcast point-level labels to all features (same as SMAP/MSL datasets)
				# Shape: (time_points,) -> (time_points, n_features)
				labels = np.tile(labels.reshape(-1, 1), (1, test.shape[1]))
				
				np.save(f'{folder}/{machine_id}_labels.npy', labels)
			else:
				# Create zero labels if label file not found
				print(f"Warning: Label file not found for {machine_id}, creating zero labels")
				labels = np.zeros_like(test)
				np.save(f'{folder}/{machine_id}_labels.npy', labels)
		
		print(f"CTF preprocessing complete: {len(machines)} machines processed")
	else:
		raise Exception(f'Not Implemented. Check one of {datasets}')

if __name__ == '__main__':
	commands = sys.argv[1:]
	load = []
	if len(commands) > 0:
		for d in commands:
			load_data(d)
	else:
		print("Usage: python preprocess.py <datasets>")
		print(f"where <datasets> is space separated list of {datasets}")