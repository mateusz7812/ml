import numpy as np


class Perceptron(object):
	"""Klasyfikator - perceptron.

		Parametry
		------------
		eta : zmiennoprzecinkowy
			Współczynnik uczenia (w przedziale pomiędzy 0.0 a 1.0)
		n_iter : liczba całkowita
			Liczba przebiegów po zestawach uczących.

		Atrybuty
		-----------
		w_ : jednowymiarowa tablica
			Wagi po dopasowaniu.
		errors_ : lista
			Liczba nieprawidłowych klasyfikacji w każdej epoce.

		"""
	def __init__(self, eta=0.01, n_iter=10, random_seed=1):
		self.eta = eta
		self.n_iter = n_iter
		self.random_seed = random_seed
	def fit(self, X, y):
		"""Dopasowanie danych uczących.

			Parametry
			----------
			X : {tablicopodobny}, wymiary = [n_próbek, n_cech]
				Wektory uczące, gdzie n_próbek
				oznacza liczbę próbek, a
				n_cech — liczbę cech.
			y : tablicopodobny, wymiary = [n_próbek]
				Wartości docelowe.

			Zwraca
			-------
			self : obiekt

			"""
		"""self.w_ = np.zeros(1 + X.shape[1])"""
		rgen = np.random.RandomState(self.random_seed)
		self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+ X.shape[1])
		self.errors_ = []

		for _ in range(self.n_iter):
			errors = 0
			for xi, target in zip(X, y):
				update = self.eta * (target - self.predict(xi))
				self.w_[1:] += update * xi
				self.w_[0] += update
				errors += int(update != 0.0)
			self.errors_.append(errors)
		return self

	def net_input(self, X):
		"""Oblicza całkowite pobudzenie"""
		return np.dot(X, self.w_[1:]) + self.w_[0]

	def predict(self, X):
		"""Zwraca etykietę klas po obliczeniu funkcji skoku jednostkowego"""
		return np.where(self.net_input(X) >= 0.0, 1, -1)