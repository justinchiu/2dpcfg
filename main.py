from rich.progress import track

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp as lse

import optax
import equinox as eqx

import matplotlib.pyplot as plt

num_classes = 8

key = jax.random.PRNGKey(1)
X0 = jax.random.randint(key, (100,2,2), 0, num_classes)
X0 = jnp.concatenate((X0, X0), 1)
X = jnp.concatenate((X0, X0), 2)

print(X[0])

#plt.imshow(X[0])
#plt.show()

def cky(params, obs):
    emit, rules = params
    width, height = obs.shape
    classes, states = emit.shape
    s, _, _, _, _ = rules.shape
    assert s == states
    assert len(rules.shape) == 5

    emit = jnp.exp(emit - lse(emit, 0, keepdims=True))
    rules = jnp.exp(rules
        - lse(rules.reshape(states, -1), -1).reshape((states,) + 4*(1,)))

    chart = jnp.zeros((height, height+1, width, width+1, states))

    for x in jnp.arange(width+1):
        for y in jnp.arange(height+1):
            chart = chart.at[x, x+1, y, y+1].set(emit[obs[x, y]])

    for w in jnp.arange(2, width+2):
        for h in jnp.arange(2, height+2):
            for x in jnp.arange(width-w+2):
                for y in jnp.arange(height-h+2):
                    for sx in jnp.arange(1,w):
                        for sy in jnp.arange(1,h):
                            #print((x.item(), (x+sx).item(), (x+w).item()),
                                #(y.item(), (y+sy).item(), (y+h).item()))
                            term = jnp.einsum(
                                "a,b,c,d,sabcd->s",
                                chart[x,    x+sx, y,    y+sy],
                                chart[x,    x+sx, y+sy, y+h],
                                chart[x+sx, x+w,  y,    y+sy],
                                chart[x+sx, x+w,  y+sy, y+h],
                                rules,
                            )
                            chart = chart.at[x,x+w, y,    y+h,:].add(term)
    return chart[0,-1,0,-1].sum()

emit = jax.random.uniform(key, 2*(num_classes,), minval=-0.01, maxval=0.01)
rules = jax.random.uniform(key, 5*(num_classes,), minval=-0.01, maxval=0.01)
params = (emit, rules)

def loss(params, X):
    return jax.vmap(cky, (None, 0), 0)(params, X).mean()
#dcky = jax.jit(jax.value_and_grad(loss))
dcky = jax.value_and_grad(loss)


optimizer = optax.adam(1e-3)
# Obtain the `opt_state` that contains statistics for the optimizer.
opt_state = optimizer.init(params)

for i in track(range(1000)):
    Z, grads = dcky(params, X)
    import pdb; pdb.set_trace()
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
import pdb; pdb.set_trace()
