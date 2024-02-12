# Name: Siyuan Huang 
# PennKey: hsy2001
import numpy as np


class HistogramFilter(object):
    """
    Class HistogramFilter implements the Bayes Filter on a discretized grid space.
    """

    def histogram_filter(self, cmap, belief, action, observation):
        '''
        Takes in a prior belief distribution, a colormap, action, and observation, and returns the posterior
        belief distribution according to the Bayes Filter.
        :param cmap: The binary NxM colormap known to the robot.
        :param belief: An NxM numpy ndarray representing the prior belief.
        :param action: The action as a numpy ndarray. [(1, 0), (-1, 0), (0, 1), (0, -1)]
        :param observation: The observation from the color sensor. [0 or 1].
        :return: The posterior distribution.
        '''

        ### Your Algorithm goes Below.
        a_k = np.zeros_like(belief)

        ## Construct \sum_{x'} a_k T_{x'x}
        di = -action[1]
        dj = action[0]
        if di != 0:
            slicing_crop = tuple(
                int(abs((di - 1) / 2)) * [slice(0, -1)] \
                    + int(abs((di + 1) / 2)) * [slice(1, None)] \
                        + [slice(0, None)]
            )
            slicing_fill = tuple(
                int(abs((di + 1) / 2)) * [slice(0, -1)] \
                    + int(abs((di - 1) / 2)) * [slice(1, None)]\
                        + [slice(0, None)]
            )
            # print(slicing_crop, slicing_fill)
            a_k[slicing_crop] = belief[slicing_fill] * 0.9
            a_k[slicing_fill] += belief[slicing_fill] * 0.1
            a_k[-int(di > 0), :] += belief[-int(di > 0), :]
            # print(a_k)
        else:
            slicing_crop = tuple(
                [slice(0, None)] + \
                    int(abs((dj - 1) / 2)) * [slice(0, -1)] \
                        + int(abs((dj + 1) / 2)) * [slice(1, None)]
            )
            slicing_fill = tuple(
                [slice(0, None)]\
                    + int(abs((dj + 1) / 2)) * [slice(0, -1)]\
                        + int(abs((dj - 1) / 2)) * [slice(1, None)]
            )
            # print(slicing_crop, slicing_fill)
            a_k[slicing_crop] = belief[slicing_fill] * 0.9
            a_k[slicing_fill] += belief[slicing_fill] * 0.1
            a_k[:, -int(dj > 0)] += belief[:, -int(dj > 0)]
        ## Construct M
        M = np.zeros_like(cmap)
        M[np.where(cmap == observation)] = 1.0
        M = np.abs(M - 0.1) # place where observation match ground-truth: 0.9 chance of being here
        a_k = np.multiply(M, a_k)
        a_k = a_k / np.sum(a_k)
        return a_k