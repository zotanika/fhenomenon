# L2 — storage. What a polynomial and a ciphertext are, and the residues they
# are bound to.
#
# Two halves. Layout.cpp is the value half: shapes as index arithmetic, with
# no memory behind them. PolySlab.cpp, RnsPoly.cpp and Ciphertext.cpp are the
# binding half: an owner that knows bytes and nothing else, non-owning
# const/mutable handles a kernel takes, and the one owner that carries a
# capacity separately from its live shape.
#
# DEPENDS fhn_ckks_params and nothing else. Not L1: a layout describes residues
# but never transforms them, and a dependency here is exactly how a
# `poly.toNtt(tables)` method would arrive — which is invariant I3 failing.
# Not L3 either, and that one is mechanical: the Arena is layer 3, above this,
# so "arena-allocated RnsPoly" is not a concept this layer can express. Which
# allocation a polynomial's words came from is a fact about an L4 call site,
# the one layer that sees both — and the handles here are what make that call
# site unable to tell the difference.
fhn_ckks_layer(fhn_ckks_storage LAYER 2
  SOURCES Layout.cpp PolySlab.cpp RnsPoly.cpp Ciphertext.cpp
  DEPENDS fhn_ckks_params)
