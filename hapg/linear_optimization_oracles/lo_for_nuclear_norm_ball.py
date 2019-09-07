from fbpca import pca
class LONuclearNormBall:
    def __init__(self,
                 radius=1
                 ):
        self.radius = radius

    def lo_oracle(self, target_matrix):
        """
        Y = argmax_{||Y||_*<=self.radius} X'*Y
        :param target_matrix: ndarray
        :return:
        Y: ndarray
        """
        (U, _, Va) = pca(target_matrix, 1, True)

        return self.radius*U*Va


