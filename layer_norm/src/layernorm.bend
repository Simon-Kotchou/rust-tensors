# Custom data types for tensors
data Tensor:
  Matrix { rows: List[Tensor], shape: List[U24] }
  Vector { elems: List[F24], length: U24 }
  Scalar { value: F24 }

# Epsilon constant to avoid division by zero
def EPS:
  return 0.00001

# Computes the sum of elements in a Tensor
def sum(xs: Tensor) -> F24:
  match xs:
    case Tensor/Matrix:
      return fold_tensor(xs, lambda t, acc: acc + sum(t), 0.0)
    case Tensor/Vector:
      return fold_tensor(xs, lambda x, acc: acc + x, 0.0)
    case Tensor/Scalar:
      return xs.value

# Computes the mean of elements in a Tensor  
def mean(xs: Tensor) -> F24:
  match xs:
    case Tensor/Matrix:
      let len = prod(xs.shape)
      return sum(xs) / len
    case Tensor/Vector:
      return sum(xs) / xs.length
    case Tensor/Scalar:
      return xs.value

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

# Performs layernorm on a Matrix Tensor 
def layernorm(xs: Tensor, w: Tensor, b: Tensor) -> Tensor:
  match xs:
    case Tensor/Matrix:
      let rows = map(lambda row: normalize(row, w, b), xs.rows)
      return Tensor/Matrix{ rows: rows, shape: xs.shape }
    case _:
      return normalize(xs, w, b)

# Helper functions

def fold_tensor(xs: Tensor, f: (F24, F24) -> F24, init: F24) -> F24:
  match xs:
    case Tensor/Matrix:
      return fold(xs.rows, lambda row, acc: f(fold_tensor(row, f, init), acc), init)
    case Tensor/Vector:
      return fold(xs.elems, f, init)
    case Tensor/Scalar:
      return f(xs.value, init)

def map(f: F24 -> F24, xs: Tensor) -> Tensor:
  match xs:
    case Tensor/Matrix:
      let rows = map(lambda row: map(f, row), xs.rows)
      return Tensor/Matrix{ rows: rows, shape: xs.shape }
    case Tensor/Vector:
      let elems = map(f, xs.elems)
      return Tensor/Vector{ elems: elems, length: xs.length }
    case Tensor/Scalar:
      return Tensor/Scalar{ value: f(xs.value) }

def zip_with_index(f: (F24, U24) -> F24, xs: Tensor) -> Tensor:
  match xs:
    case Tensor/Matrix:
      let rows = map(lambda row, i: zip_with_index(f, row), xs.rows)
      return Tensor/Matrix{ rows: rows, shape: xs.shape }
    case Tensor/Vector:
      let elems = zip_with_index(f, xs.elems)
      return Tensor/Vector{ elems: elems, length: xs.length }
    case Tensor/Scalar:
      return Tensor/Scalar{ value: f(xs.value, 0) }

def get(xs: Tensor, i: U24) -> F24:
  match xs:
    case Tensor/Matrix:
      let row = i / xs.shape[1]
      let col = i % xs.shape[1]
      return get(xs.rows[row], col)
    case Tensor/Vector:
      return xs.elems[i]
    case Tensor/Scalar:
      return xs.value