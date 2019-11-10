class FeatureSampleNetwork():
	'''
	The FeatureSampleNewtork object contains an
	adjacent matrix, where vertices are the examples
	and the features.

	Parameters
	----------

	data : pandas.Dataframe
		A pandas Dataframe with column names and index.
		The values must be binary vectors.

	labels : pandas.Dataframe
		A pandas Dataframe with two columns: first the id,
		and second the labels.

	Attributes
	----------

	data : pandas.Dataframe
		This is where we store data
	data_label: pandas.Dataframe
		This is where we store labels
	adjacent_matrix : scipy.csr_matrix
		A sparse matrix of feature-sample network.
	'''

	from scipy import sparse
	import pandas as pd
	import numpy as np
	import warnings

	def __init__(self, data, labels):
		self.data = data
		self.data_label = labels
		self.adjacent_matrix = self._make_adjacent_matrix(data, labels)
		self.fit_flag = False

	def _make_adjacent_matrix(self, data, labels):
		'''
		This method is used to generate the sparse matrix
		of the feature-sample network.

		Parameters
		----------
		df : pandas.Dataframe
			A pandas Dataframe with column names and index.
			The values must be binary vectors.
		labels: pandas.Dataframe
			A pandas Dataframe with the elements which has label.

		Returns
		-------
		data_crs
			A sparse adjacent matrix
		'''
		df = self.pd.concat([data, labels], axis = 1).sort_values(labels.columns[0])
		df = df.drop(columns = [labels.columns[0]])
		self.processed_data = df

		index = self.np.arange(len(df.index.values), dtype = 'float32')
		features = self.np.arange(len(df.index.values),
		 		   len(df.columns)+len(df.index.values), dtype = 'float32')

		len_index = len(index)
		len_features = len(features)

		data_not_T = df.values.flatten()
		data_T = (df.values).T.flatten()

		matrix_row_index_data   = self.np.repeat(index, len_features)
		matrix_row_index_data_T = self.np.repeat(features, len_index)
		matrix_col_index_data   = self.np.tile(features, len_index)
		matrix_col_index_data_T = self.np.tile(index, len_features)

		row_sparse = self.np.concatenate((matrix_row_index_data,
									matrix_row_index_data_T))
		col_sparse = self.np.concatenate((matrix_col_index_data,
									matrix_col_index_data_T))

		data_sparse = self.np.concatenate((data_not_T, data_T))
		data_csr = self.sparse.csr_matrix((data_sparse, (row_sparse, col_sparse)),
									 	   shape = (len_index + len_features, len_index + len_features),
									 	   dtype = 'float32')
		return data_csr

	def fit(self, beta):
		'''
		This method is used to calculate the class pertinence
		level.

		Parameters
		----------
		beta : int or float
			The importance value of the labeled examples.

		Returns
		-------
		w
			A sparse adjacent matrix scaled by beta.
		p
			A sparse transition matrix.
		pertinence level
			positive-class pertinence level of each unlabeled
			example
		'''
		self.warnings.filterwarnings("ignore")
		adjacent_matrix = self.adjacent_matrix
		shape_adj_matrix = adjacent_matrix.shape
		labels_index = self.np.arange(self.data_label.shape[0])
		all_examples_index = self.np.arange(shape_adj_matrix[0])

		matrix_row_index_data   = self.np.repeat(labels_index, len(all_examples_index))
		matrix_row_index_data_T = self.np.repeat(all_examples_index, len(labels_index))
		matrix_col_index_data   = self.np.tile(all_examples_index, len(labels_index))
		matrix_col_index_data_T = self.np.tile(labels_index, len(all_examples_index))
		row_sparse = self.np.concatenate((matrix_row_index_data,
		                            matrix_row_index_data_T))
		col_sparse = self.np.concatenate((matrix_col_index_data,
                            		matrix_col_index_data_T))

		data_sparse = self.np.ones(len(matrix_col_index_data)) * (beta - 1)

		print('Gerando a matriz com os pesos...')

		w = self.sparse.csr_matrix((data_sparse, (matrix_row_index_data, matrix_col_index_data)),
									 	   shape = shape_adj_matrix,
									 	   dtype = 'float32')

		w = w.multiply(adjacent_matrix) + adjacent_matrix
		w.setdiag(1)
		self.w = w

		print('Calculando o espectro da matriz...')

		eigvalues_w, eigvectors_w = self.sparse.linalg.eigsh(w, k = 1)
		leading_eigvalue_w = eigvalues_w[0]
		print(leading_eigvalue_w)
		leading_eigvector_w = eigvectors_w[:, 0]
		print('Normalizando a matriz...')

		p = w.multiply(leading_eigvector_w)
		p = p.T
		p = p.multiply(1/(leading_eigvector_w * leading_eigvalue_w)).T
		self.p = p

		print('Calculando o espectro da matriz normalizada...')
		eigvalues_p, eigvectors_p = self.sparse.linalg.eigsh(p.T, k = 1, v0 = self.np.ones(shape_adj_matrix[0]))
		leading_eigvector_p = self.np.absolute(eigvectors_p[:, 0])
		self.pertinence_level = leading_eigvector_p/leading_eigvector_p.sum()
# 		self.pertinence_level = leading_eigvector_p
		self.fit_flag = True
		print('OK')

	def get_pertinence_unlabeled(self):
		if not self.fit_flag:
			print('Você ainda não aplicou .fit')
			return
		else:
			len_label = self.data_label.shape[0]
			len_features = len(self.data.columns)
			return self.pertinence_level[len_label:-len_features]

	def get_pertinence_features(self):
		if not self.fit_flag:
			print('Você ainda não aplicou .fit')
			return
		else:
			len_label = self.data_label.shape[0]
			len_features = len(self.data.columns)
			return self.pertinence_level[-len_features:]

	def get_pertinence_labeled(self):
		if not self.fit_flag:
			print('Você ainda não aplicou .fit')
			return
		else:
			len_label = self.data_label.shape[0]
			len_features = len(self.data.columns)
			return self.pertinence_level[:len_label]

	def get_index_labeled(self):
		len_label = self.data_label.shape[0]
		return self.processed_data.index[:len_label]

	def get_index_unlabeled(self):
		len_label = self.data_label.shape[0]
		return self.processed_data.index[len_label:]