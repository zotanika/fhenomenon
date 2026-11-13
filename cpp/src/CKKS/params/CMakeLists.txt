# L0 — the parameter value type. Depends on nothing, by design: a parameter
# set that needed anything else could not be the identity key of a
# precomputation table at L1.
fhn_ckks_layer(fhn_ckks_params LAYER 0 SOURCES Params.cpp)
