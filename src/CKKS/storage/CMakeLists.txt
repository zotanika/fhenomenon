# L2 — storage. What a polynomial and a ciphertext are, with no statement about
# where their residues live.
#
# DEPENDS fhn_ckks_params and nothing else. Not L1: a layout describes residues
# but never transforms them, and a dependency here is exactly how a
# `poly.toNtt(tables)` method would arrive — which is invariant I3 failing.
# Not L3 either, and that one is mechanical: the Arena is layer 3, above this,
# so "arena-allocated RnsPoly" is not a concept this layer can express. Which
# allocation a polynomial's words came from is a fact about an L4 call site,
# the one layer that sees both.
fhn_ckks_layer(fhn_ckks_storage LAYER 2 SOURCES Layout.cpp DEPENDS fhn_ckks_params)
