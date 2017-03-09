# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 14:19:00 2016

Module:
    birg - Bipartite Random Graph

Author:
    Mika Straka

Description:
    Implementation of the Bipartite Random Graph model (BiRG).

    Given a biadjacency matrix of an unweighted bipartite graph in the form of
    a binary array as input, the module allows the user to create the
    corresponding Bipartite Random Graph null model. The user can calculate and
    save the p-values of the observed :math:`\\Lambda`-motifs of the two
    distinct bipartite node sets, which correspond to the row and column
    indices of the biadjacency matrix.

Usage:
    Be ``mat`` a two-dimensional binary NumPy array. The nodes of the two
    bipartite layers are ordered along the columns and rows, respectively. In
    the algorithm, the two layers are identified by the boolean values ``True``
    for the **row-nodes** and ``False`` for the **column-nodes**.

    Import the module and initialize the Bipartite Random Graph model::

        >>> from src.birg import BiRG 
        >>> rg = BiRG(bin_mat=mat) 

    In order to analyze the similarity of the **row-layer nodes** and to save
    the p-values of the corresponding :math:`\\Lambda`-motifs, i.e. of the
    number of shared neighbors [Saracco2016]_, use::

        >>> rg.lambda_motifs(bip_set=True, filename=<filename>, delim='\\t',
                binary=True)

    For the **column-layer nodes**, use::

        >>> cm.lambda_motifs(bip_set=False, filename=<filename>, delim='\\t',
                binary=True)

    ``bip_set`` selects the bipartite node set for which the p-values should be
    calculated and saved. By default, the file is saved as a binary NumPy file
    ``.npy``. In order to save it as a human-readable CSV format, set
    ``binary=False`` in the function call.
    The filename *<filename>* should contain a relative path declaration. The
    default name of the output file is *p_values_<bip_set>* and ends with
    ``.npy`` or ``.csv`` depending on the variable ``binary``. In the CSV
    format, the values are separated by tabs, which can be changed using the
    ``delim`` keyword.

    Subsequently, the p-values can be used to perform a multiple hypotheses
    testing and to obtain statistically validated monopartite projections
    [Saracco2016]_. The p-values are calculated in parallel by default.

Reference:
    R. Albert, A.-L. Barabási, Statistical mechanics of complex networks
    Rev. Mod. Phys. 74, 47
    doi:http://dx.doi.org/10.1103/RevModPhys.74.47

    [Saracco2016] F. Saracco, M. J. Straka, R. Di Clemente, A. Gabrielli, G.
    Caldarelli, T. Squartini, Inferring monopartite projections of bipartite
    networks: an entropy-based approach, arXiv preprint arXiv:1607.02481
"""

import os
import numpy as np
from scipy.stats import binom


class BiRG:
    """Bipartite Random Graph for undirected binary bipartite networks.

    This class implements the Bipartite Random Graph (BiGM), which can be used
    as a null model for the analysis of undirected and binary bipartite
    networks. The class provides methods to calculate the biadjacency matrix of
    the null model and to quantify node similarities in terms of p-values.  
    """

    def __init__(self, bin_mat):
        """Initialize the parameters of the BiRG.

        :param bin_mat: binary input matrix describing the biadjacency matrix
                of a bipartite graph with the nodes of one layer along the rows
                and the nodes of the other layer along the columns.
        :type bin_mat: numpy.array
        """
        self.bin_mat = np.array(bin_mat)
        self.check_input_matrix_is_binary()
        [self.num_rows, self.num_columns] = self.bin_mat.shape
        self.num_edges = self.get_number_edges()
        self.edge_prob = self.get_edge_prob()
        self.lambda_prob = self.get_lambda_motif_prob()

# ------------------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------------------

    def check_input_matrix_is_binary(self):
        """Check that the input matrix is binary, i.e. entries are 0 or 1.

        :raise AssertionError: raise an error if the input matrix is not
            binary
        """
        assert np.all(np.logical_or(self.bin_mat == 0, self.bin_mat == 1)), \
            "Input matrix is not binary."

    def get_number_edges(self):
        """Return the number of edges encoded in the biadjacency matrix.
        
        :returns: number of edges in the graph
        :rtype: float
        """
        # edges are indicated as "1" in the matrix
        return self.bin_mat.sum()

    def get_edge_prob(self):
        """Return the uniform edge probability of the Bipartite Random Graph.
        
        :returns: edge probability
        :rtype: float
        """
        p = float(self.num_edges) / (self.num_rows * self.num_columns)
        return p

    def get_lambda_motif_prob(self):
        """Return the probability of a :math:`\\Lambda`-motif. 
        
        For two nodes nodes :math:`i, j`, the probability of the motif
        :math:`\\Lambda_{ij}^{\\alpha}` is given by

        .. math::

            p(\\Lambda_{ij}^{\\alpha}) = p_{i\\alpha} * p_{j\\alpha}.

        :returns: probability for a :math:`\\Lambda`-motif
        :rtype: float
        """
        pp = self.edge_prob * self.edge_prob
        return pp

# ------------------------------------------------------------------------------
# Total log-likelihood of the observed Lambda motifs in the input matrix
# ------------------------------------------------------------------------------

    def lambda_loglike(self, bip_set=False):
        """Return the maximum likelihood of the edge weights of the projection
        on the specified bipartite set.

        :param bip_set: analyze countries (True) or products (False)
        :type bip_set: bool
        :param write: if True, the pvalues are saved in an external file
        :type write: bool
        """
        lambda_mat = self.get_lambda_vals(bip_set)
        p_mat = self.get_proj_pmat(lambda_mat, bip_set)
        logp = np.log(p_mat[np.triu_indices_from(p_mat, k=1)])
        loglike = logp.sum()
        return loglike

    def get_proj_pmat(self, mat, bip_set=False):
        """Return a matrix of probabilites for the observed Lambda_2 motifs in
        the input matrix.

            pmf(k) = Pr(X = k)

        The lower triangular part (including the diagonal) of the pvalue
        matrix is set to zero.

        :param bip_set: selects countries (True) or products (False)
        :type bip_set: bool
        :param mat: matrix of observed Lambda_2 motifs
        :type mat: np.array
        """
        bn = self.get_binomial(bip_set)
        m = bn.pmf(mat)
        m[np.tril_indices_from(m, k=0)] = 0
        return m

# ------------------------------------------------------------------------------
# Lambda motifs
# ------------------------------------------------------------------------------

    def lambda_motifs(self, bip_set=False, filename=None, delim='\t',
                            binary=True):
        """Calculate and save the p-values of the :math:`\\Lambda`-motifs.

        For each node couple in the bipartite layer specified by ``bip_set``,
        the p-values of the corresponding :math:`\\Lambda`-motifs are
        calculated based on the biadjacency matrix of the BiRG null model.

        The results can be saved either as a binary or a human-readable file.

        .. note::

            * The output consists of one array of p-values to keep memory usage
            low. If the bipartite layer ``bip_set`` contains ``n`` nodes,
            this means that the array will contain :math:`\\binom{n}{2}``
            entries. The indices of the nodes corresponding to entry ``k``
            in the array can be reconstructed using the method
            ``flat2_triumat_idx(k, n)``.
            * If ``binary == False``, the ``filename`` should end with
            ``.csv``. Otherwise, it will be saved in binary format and the
            suffix ``.npy`` will be appended automatically. By default, the file
            is saved in binary format.
                
        :param bip_set: select row-nodes (``True``) or column-nodes (``False``)
        :type bip_set: bool
        :param parallel: select whether the calculation of the p-values should
            be run in parallel (``True``) or not (``False``)
        :type parallel: bool
        :param filename: name of the file which will contain the p-values
        :type filename: str
        :param delim: delimiter between entries in file, default is tab
        :type delim: str
        :param binary: if ``True``, the file will be saved in the binary 
            NumPy format ``.npy``, otherwise as ``.csv``
        :type binary: bool
        """
        lambda_mat = self.get_lambda_vals(bip_set)
        pval_mat = self.get_lambda_pvalues(lambda_mat, bip_set)
        if filename is None:
            fname = 'p_values_' + str(bip_set)
            if not binary:
                fname +=  '.csv'
        else:
            fname = filename
        # account for machine precision:
        pval_mat += np.finfo(np.float).eps
        self.save_matrix(pval_mat, filename=fname, delim=delim, binary=binary)

    def get_lambda_vals(self, bip_set=False):
        """Return an array of observed :math:`\\Lambda`-motifs.

       The number of elements in the returned array ``A`` is
       :math:`\\binom{N}{2}`, where :math:`N` is the number of distinct nodes
       in the bipartite layer ``bip_set``. The entries are given as

        .. math::

            A_{ij} = N(\\Lambda_{ij})

        :param bip_set: select row-nodes (``True``) or column-nodes (``False``)
        :type bip_set: bool

        :returns: array of observed :math:`\\Lambda`-motifs
        :rtype: numpy.array

        :raise NameError: raise an error if the parameter ``bip_set`` is
            neither ``True`` nor ``False``
        """
        if not bip_set:
            lambda_mat = np.dot(np.transpose(self.bin_mat), self.bin_mat)
            assert lambda_mat.shape == (self.num_columns, self.num_columns)
        elif bip_set:
            lambda_mat = np.dot(self.bin_mat, np.transpose(self.bin_mat))
            assert lambda_mat.shape == (self.num_rows, self.num_rows)
        else:
            errmsg = str(bip_set) + 'not supported.'
            raise NameError(errmsg)
        # set diagonal to zero
#        di = np.diag_indices(lambda_mat.shape[0], 2)
#        lambda_mat[di] = 0
        return lambda_mat[np.triu_indices_from(lambda_mat, k=1)]

    def get_lambda_pvalues(self, mat, bip_set=False):
        """Return the p-values of the observed :math:`\\Lambda`-motifs.

        The p-values are defined as

        .. math::

            p_{value}(k) = Pr(X >= k) = 1 - cdf(k) + pmf(k),
        
        with :math:`cdf(k) = Pr(X <= k)` being the cumulative distribution 
        function and :math:`pmf(k) = Pr(X = k)` the probability mass function.

        The number of p-values in the returned array is :math:`\\binom{N}{2}`,
        where :math:`N` is the number of distinct nodes in the bipartite layer
        ``bip_set``.

        :param mat: matrix of observed :math:`\\Lambda`-motifs
        :type mat: numpy.array
        :param bip_set: select row-nodes (``True``) or column-nodes (``False``)
        :type bip_set: bool

        :returns: array with p-values
        :rtype: numpy.array
        """
        bn = self.get_binomial(bip_set)
        m = 1 - bn.cdf(mat) + bn.pmf(mat)
#        m[np.tril_indices_from(m, k=0)] = 0
        return m

    def get_binomial(self, bip_set=False):
        """Return a binomial probability distribution.
        
        The distribution is obtained from the probability of observing
        :math:`\\Lambda`-motifs, :math:`P(\\Lambda_{ij}^{\\alpha}` and the
        number of elements in the opposite layer of ``bip_set``.  
        
        :param bip_set: select row-nodes (``True``) or column-nodes (``False``)
        :type bip_set: bool

        :returns: binomial probability distribution
        :rtype: scipy.stats.binom instance

        :raise NameError: raise an error if the parameter ``bip_set`` is
            neither ``True`` nor ``False``
        """
        if not bip_set:
            n = self.num_rows
        elif bip_set:
            n = self.num_columns
        else:
            errmsg = str(bip_set) + 'not supported.'
            raise NameError(errmsg)
        bn = binom(n, self.lambda_prob)
        return bn

# ------------------------------------------------------------------------------
# Auxiliary methods
# ------------------------------------------------------------------------------

    def get_triup_dim(self, bip_set):
        """Return the number of possible couples in ``bip_set``.
        
        :param bip_set: selects row-nodes (``True``) or column-nodes (``False``)
        :type bip_set: bool

        :returns: return the number of node couple combinations corresponding 
        to the layer ``bip_set
        :rtype: int 

        :raise NameError: raise an error if the parameter ``bip_set`` is
            neither ``True`` nor ``False``
        """
        if bip_set:
            return self.num_rows * (self.num_rows - 1) / 2
        elif not bip_set:
            return self.num_columns * (self.num_columns - 1) / 2
        else:
            errmsg = "'" + str(bip_set) + "' " + 'not supported.'
            raise NameError(errmsg)

    @staticmethod
    def triumat2flat_idx(i, j, n):
        """Convert an matrix index couple to a flattened array index.

        Given a square matrix of dimension :math:`n` and an index couple
        :math:`(i, j)` *of the upper triangular part* of the matrix, the
        function returns the index which the matrix element would have in a
        flattened array.

        .. note::
            * :math:`i \\in [0, ..., n - 1]`
            * :math:`j \\in [i + 1, ..., n - 1]`
            * returned index :math:`\\in [0,\\, n (n - 1) / 2 - 1]`

        :param i: row index
        :type i: int
        :param j: column index
        :type j: int
        :param n: dimension of the square matrix
        :type n: int

        :returns: flattened array index
        :rtype: int
        """
        return int((i + 1) * n - (i + 2) * (i + 1) / 2. - (n - (j + 1)) - 1)

    def flat2triumat_idx(self, k, n):
        """Convert an array index to the index couple of a triangular matrix.

        ``k`` is the index of an array of length :math:`frac{n(n - 1)}{2}`,
        which contains the elements of an upper triangular matrix of dimension
        ``n`` excluding the diagonal. The function returns the index couple
        :math:`(i, j)` that corresponds to the entry ``k`` of the flat array.

        .. note::
            * :math:`k \\in [0,\\, n (n - 1) / 2 - 1]`
            * returned indices:
                * :math:`i \\in [0, ..., n - 1]`
                * :math:`j \\in [i + 1, ..., n - 1]`

        :param k: flattened array index
        :type k: int
        :param n: dimension of the square matrix
        :type n: int

        :returns: matrix index tuple (row, column)
        :rtype: tuple
        """
        i = self.get_row_idx(k, n)
        j = self.get_column_idx(k, n, i)
        return (i, j)

    @staticmethod
    def save_matrix(mat, filename, delim='\t', binary=False):
        """Save the matrix ``mat`` in the file ``filename``.

        The matrix can either be saved as a binary NumPy ``.npy`` file or as a
        human-readable CSV file.

        .. note:: The relative path has to be provided in the filename, e.g.
                *../data/pvalue_matrix.csv*

        :param mat: two-dimensional matrix
        :type mat: numpy.array
        :param filename: name of the output file
        :type filename: str
        :param delim: delimiter between values in file
        :type delim: str
        :param binary: if ``True``, save as binary ``.npy``, otherwise as a
            CSV file
        :type binary: bool
        """
        if binary:
            np.save(filename, mat)
        else:
            np.savetxt(filename, mat, delimiter=delim)

#    def save_pvalues(self, pval_mat, bip_set=False):
#        if not bip_set:
#            b = 'products'
#        elif bip_set:
#            b = 'countries'
#        else:
#            errmsg = "'" + str(bip_set) + "' " + 'not supported.'
#            raise NameError(errmsg)
#        s = ['./data/BiRG_data/', 'hs2007_BiRG_P_', b, '_',
#                '.csv'] 
#        filename = ''.join(s)
#        np.savetxt(filename, pval_mat, delimiter='\t')

################################################################################
# Main
################################################################################

if __name__ == "__main__":
    pass
