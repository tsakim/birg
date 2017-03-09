# Bipartite Random Graph for Python

The Bipartite Random Graph (BiRG) is a statistical null model for binary bipartite networks. It offers an unbiased method of analyzing node similarities and obtaining statistically validated monopartite projections \[Saracco2016\].

The BiRG is an extension of the Erdős–Rényi model \[ER\] to bipartite graphs.
Link probabilities are obtained by constraining the total number of links
between the two layers in the network. Similar null-models, which are based on
the principle of entropy maximization, are 

* [BiCM](https://github.com/tsakim/bicm) - Bipartite Configuration Model
* [BiPCM](https://github.com/tsakim/bipcm) - Bipartite Partial Configuration Model

Please consult the original articles for details about the underlying methods
and applications to user-movie and international trade databases \[Saracco2016,
Straka2016\].  

## Author 

Mika J. Straka

## Version and Documentation

The newest version of the module can be found on
[https://github.com/tsakim/birg](https://github.com/tsakim/birg).

## How to cite

If you use the `birg` module, please cite its location on Github
[https://github.com/tsakim/birg](https://github.com/tsakim/birg) and the
original article \[Saracco2016\]. 

### References

\[ER\] [P. Erdős, A. Rényi, On Random Graphs. I, Publicationes
Mathematicae, 6, 290–297 (1959)](http://www.renyi.hu/~p_erdos/1959-11.pdf)

\[Saracco2016\] [F. Saracco, M. J. Straka, R. Di Clemente, A. Gabrielli, G. Caldarelli, T. Squartini, Inferring monopartite projections of bipartite networks: an entropy-based approach, arXiv preprint arXiv:1607.02481](https://arxiv.org/abs/1607.02481)

\[Straka2016\] [M. J. Straka, F. Saracco, G. Caldarelli, Product Similarities in International Trade from Entropy-based Null Models, Complex Networks 2016, 130-132 (11 2016), ISBN 978-2-9557050-1-8](http://www.complexnetworks.org/BookOfAbstractCNA16.pdf)

---
Copyright (c) 2015-2017 Mika J. Straka 
