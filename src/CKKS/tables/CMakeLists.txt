# L1 — derived precomputation, and the transforms that consume it.
#
# DEPENDS is deliberately empty. Tables are keyed by what they precompute
# (modulus, log_degree), not by "the session" and not by a parameter set, so
# this layer depends on nothing at all — not even L0. That is what lets
# CkksNttTest run with no Params, no keys and no arena.
#
# The negacyclic transform lives here rather than at L4 because L4 is eval::,
# the layer of CKKS operations over ciphertexts and keys. A transform over one
# prime has none of those inputs and needs no scratch (in-place Cooley-Tukey
# and Gentleman-Sande butterflies allocate nothing), so placing it above would
# force this layer's test to reach upward for something it does not use.
fhn_ckks_layer(fhn_ckks_tables LAYER 1 SOURCES NttTables.cpp)
