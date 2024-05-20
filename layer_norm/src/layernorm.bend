def EPS:
  return 0.00001

# Custom data types for tensors
type Tensor = Tensor { shape: List[U24], data: List[F24] }

# Computes the sum of elements in a Tensor
def sum(xs: Tensor) -> F24:
  return fold(xs.data, lambda x, acc: acc + x, 0.0)

# Computes the mean of elements in a Tensor  
def mean(xs: Tensor) -> F24:
  let len = prod(xs.shape)
  return sum(xs) / len

# Computes the variance of elements in a Tensor
def variance(xs: Tensor, m: F24) -> F24:
  let diff = map(lambda x: pow(x - m, 2.0), xs)
  return mean(diff)

# Normalizes a Tensor using layernorm
def normalize(xs: Tensor, w: Tensor, b: Tensor) -> Tensor:
  let m = mean(xs)
  let v = variance(xs, m)
  let rstd = pow(v + EPS, -0.5)
  
  let f = lambda x, i:
    let wi = get(w, i)
    let bi = get(b, i)
    (x - m) * rstd * wi + bi
  
  return zip_with_index(f, xs)

# Performs layernorm on a Tensor 
def layernorm(xs: Tensor, w: Tensor, b: Tensor) -> Tensor:
  return normalize(xs, w, b)

# Helper functions

def prod(xs: List[U24]) -> U24:
  return fold(xs, lambda x, acc: x * acc, 1)

def map(f: F24 -> F24, xs: Tensor) -> Tensor:
  let data = map(f, xs.data)
  return Tensor{ shape: xs.shape, data: data }

def zip_with_index(f: (F24, U24) -> F24, xs: Tensor) -> Tensor:
  let data = zip_with_index(f, xs.data)
  return Tensor{ shape: xs.shape, data: data }

def get(xs: Tensor, i: U24) -> F24:
  return xs.data[i]

# Random number generator (xorshift32)  
def rand(seed: U24) -> U24:
  let seed = seed ^ (seed << 13) 
  let seed = seed ^ (seed >> 17)
  let seed = seed ^ (seed << 5)
  return seed

# Generates a random F24 number between 0 and 1  
def randf(seed: U24) -> F24:  
  return (rand(seed) % 0xFFFFFF) / 0xFFFFFF

# Generates a random tensor
def rand_tensor(seed: U24, shape: List[U24]) -> Tensor:
  let size = prod(shape)
  bend i = 0, data = List/Nil:
    when i < size:
      let r = randf(seed + i)
      data = List.append(fork(i+1), r)
    else:
      data = data
  return Tensor{ shape: shape, data: data }

# Computes the mean squared error between two tensors
def mse(xs: Tensor, ys: Tensor) -> F24:
  let diff = map(lambda x, y: pow(x - y, 2.0), zip(xs.data, ys.data))
  return mean(Tensor{ shape: xs.shape, data: diff })

# Tests the correctness of layernorm on random tensors
def test_layernorm(n: U24, max_dim: U24, tolerance: F24) -> Bool:
  bend i = 0, all_passed = True:
    when i < n:
      let shape = gen_shape(i, max_dim)
      let xs = rand_tensor(i, shape)
      let w = rand_tensor(i*2, List/Cons(prod(List.tail(shape)), List/Nil))
      let b = rand_tensor(i*3, List/Cons(prod(List.tail(shape)), List/Nil))
      let ys = layernorm(xs, w, b)
      let error = mse(xs, ys)
      all_passed = all_passed && (error < tolerance)
    else:
      all_passed = all_passed
  return all_passed

# Generates a random tensor shape
def gen_shape(seed: U24, max_dim: U24) -> List[U24]:
  let rank = (rand(seed) % max_dim) + 1
  bend i = 0, shape = List/Nil:
    when i < rank:
      let dim = (rand(seed + i) % max_dim) + 1
      shape = List.append(fork(i+1), dim)
    else:
      shape = shape
  return shape

def main:
  let passed = test_layernorm(100, 4, 0.001)
  if passed:
    return "All tests passed!"
  else:
    return "Some tests failed!"