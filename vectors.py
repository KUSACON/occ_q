from math import pow
from typing import SupportsIndex


class Vector(list):
    """Implement a vector of arbitrary size and most simple mathematical operations with it.
    Usage: Vector(<iterable>) or Vector(size, default_element=0)"""

    def __init__(self, *args):
        default_el = 0
        if len(args) >= 2:
            default_el = args[1]
        if len(args) >= 1 and type(args[0]) is int:
            super().__init__([default_el for i in range(args[0])])
        else:
            super().__init__(*args)

    def __eq__(self, other):
        if len(self) != len(other):
            return False
        return all([a == b for (a, b) in zip(self, other)])

    def __ne__(self, other):
        if len(self) != len(other):
            return True
        return any([a != b for (a, b) in zip(self, other)])

    def __add__(self, other):
        """Add two vectors, provided they are of the same length"""
        if len(self) != len(other):
            raise ValueError("Vectors must have the same length!")
        return Vector([a + b for (a, b) in zip(self, other)])

    def __sub__(self, other):
        """Subtract two vectors, provided they are of the same length"""
        if len(self) != len(other):
            raise ValueError("Vectors must have the same length!")
        return Vector([a - b for (a, b) in zip(self, other)])

    def __iadd__(self, other):
        return self.__add__(other)

    def __isub__(self, other):
        return self.__sub__(other)

    def __xor__(self, other):
        """Compute dot product of two vectors"""
        if len(self) != len(other):
            raise ValueError("Vectors must have the same length!")
        return sum([(a * b) for (a, b) in zip(self, other)])

    def __mul__(self, other):
        """If other is a scalar, multiply by it; If it's a vector, compute vector product.
        If it's a matrix (or any 2d iterable), perform vector by matrix mult.
        Always returns a vector. Dot product is done by __xor__ (a ^ b)."""
        if hasattr(other, '__iter__'):
            if other and hasattr(other[0], '__iter__'):
                return Matrix(other) * self

            if len(self) != len(other):
                raise ValueError("Vectors must have the same length!")
            elif len(self) != 3:
                raise NotImplementedError("Vector product for arbitrary sized vectors is not implemented, only size 3")
            else:
                return Vector([self[1] * other[2] - self[2] * other[1],
                               self[2] * other[0] - self[0] * other[2],
                               self[0] * other[1] - self[1] * other[0]])
        else:
            return Vector([other * x for x in self])

    def __truediv__(self, other):
        """Really just multiply by 1/x where x is a scalar"""
        return Vector([x / other for x in self])

    def __imul__(self, other):
        return self.__mul__(other)

    def __itruediv__(self, other):
        return self.__truediv__(other)

    def __abs__(self):
        """Magnitude of the vector"""
        return self.length()

    def length(self):
        """Magnitude of the vector"""
        return sum(self) ** 0.5

    def normalize(self):
        return self / self.length()


class Matrix(list):
    """A vector of vectors, which are rows
    Usage: Matrix(<vector of rows>) or Matrix(rows, cols, default value=0)"""
    def __init__(self, *args):
        self.n_cols = self.n_rows = 0
        if len(args) == 1:
            _rows = args[0]
            if hasattr(_rows, '__iter__'):
                self.n_rows = len(_rows)
                for el in _rows:
                    if hasattr(el, '__iter__'):
                        if self.n_cols == 0:
                            self.n_cols = len(el)
                        elif self.n_cols != len(el):
                            raise ValueError("All rows must have the same length")
                super().__init__([Vector(el) for el in _rows])
                return

        default_el = 0
        if len(args) >= 3:
            default_el = args[2]
        if len(args) >= 2 and type(args[0]) is int and type(args[1]) is int:
            self.n_rows, self.n_cols = args[0], args[1]
            super().__init__([Vector([default_el for i in range(self.n_cols)]) for j in range(self.n_rows)])
            return

        super().__init__(*args)

    @staticmethod
    def from_cols(cols):
        """Create a matrix and initialize it with a list of columns"""
        if cols:
            n_rows = len(cols[0])
            for c in cols:
                if len(c) != n_rows:
                    raise ValueError("Columns must have the same size")
            return Matrix(cols).transpose()

    @staticmethod
    def identity_matrix(sz):
        """Create an identity matrix of given size"""
        new_matr = Matrix(sz, sz, 0.)
        for i in range(sz):
            new_matr[i][i] = 1.
        return new_matr

    def __str__(self):
        str_rows = [' ' + str(r) for r in self]
        if str_rows:
            str_rows[0] = str_rows[0].lstrip()
        out = '\n'.join(str_rows)
        return f'[{out}]'

    def __add__(self, other):
        """Perform matrix addition (must have the same size)"""
        if not isinstance(other, Matrix):
            raise ValueError("Can only add matrices")
        if self.n_rows != other.n_rows or self.n_cols != other.n_cols:
            raise ValueError("Must have the same number of rows and columns")
        return Matrix([a + b for (a, b) in zip(self, other)])

    def __sub__(self, other):
        """Perform matrix subtraction (must have the same size)"""
        if not isinstance(other, Matrix):
            raise ValueError("Can only add matrices")
        if self.n_rows != other.n_rows or self.n_cols != other.n_cols:
            raise ValueError("Must have the same number of rows and columns")
        return Matrix([a - b for (a, b) in zip(self, other)])

    def __iadd__(self, other):
        return self.__add__(other)

    def rows(self):
        """Get a list of rows (as vectors)"""
        return list(self)

    def cols(self):
        """Get a list of columns (as vectors)"""
        out = [Vector() for i in range(self.n_cols)]
        for i in range(self.n_cols):
            for j in range(self.n_rows):
                out[i].append(self[j][i])
        return out

    def transpose(self):
        """Transpose the matrix (swap rows and columns)"""
        return Matrix(self.cols())

    def take(self, rows, cols, r_from=0, c_from=0):
        """Returns a part of this matrix of the given size, starting from the given row/col"""
        rows = self[r_from:r_from+rows+r_from]
        cut_rows = [Vector(x[c_from:c_from+cols]) for x in rows]
        return Matrix(cut_rows)

    def append(self, row):
        """Takes a vector and adds it as a new row of the matrix"""
        if not hasattr(row, '__iter__'):
            raise ValueError("Appending a row; must be an iterable")
        if self.n_cols == 0:
            self.n_cols = len(row)
        if self.n_cols != len(row):
            raise ValueError("Must have the same number of elements (colums) as other rows")
        self.n_rows += 1
        super().append(row)

    def append_col(self, col):
        """Takes a vector and adds it as a new column of the matrix"""
        if not hasattr(col, '__iter__'):
            raise ValueError("Appending a column; must be an iterable")
        if self.n_rows == 0:
            for i in range(len(col)):
                self.append(Vector())
        if self.n_rows != len(col):
            raise ValueError("Must have the same number of elements (rows) as other columns")
        self.n_cols += 1
        for (r, new_el) in zip(self, col):
            r.append(new_el)

    def pop_col(self, nc):
        """Removes a column from matrix and returns it"""
        col = Vector()
        for row in self:
            col.append(row.pop(nc))
        return col

    def popped(self, nr):
        """Matrix without the i-th row"""
        new_rows = [r for (i, r) in enumerate(self.rows()) if i != nr]
        return Matrix(new_rows)

    def popped_col(self, nc):
        """Matrix without the j-th column"""
        new_cols = [c for (i, c) in enumerate(self.cols()) if i != nc]
        return Matrix.from_cols(new_cols)

    def __mul__(self, other):
        """If other is a matrix, perform matrix multiplication (returns a matrix)
           If it's a vector, perform matrix by vector multiplication (returns a vector)
           If it is a scalar, perform matrix by scalar multiplication (returns a matrix)"""
        if isinstance(other, Matrix):
            if self.n_cols != len(other):
                raise ValueError("Matrix A must have the same number of columns as matrix B has rows")
            return Matrix([[a ^ b for b in other.cols()] for a in self])
        elif isinstance(other, Vector):
            return Vector([other ^ r for r in self])
        elif hasattr(other, '__iter__'):
            if other and hasattr(other[0], '__iter__'):
                return self * Matrix(other)
            else:
                return self * Vector(other)
        else:
            return Matrix([r * other for r in self])

    def __imul__(self, other):
        return self.__mul__(other)

    def minor(self, i, j):
        """Get the minor of element M[i][j]"""
        if self.n_cols != self.n_rows:
            raise NotImplementedError("Only possible for square matrices")
        return self.popped(i).popped_col(j).det()

    def minor_matrix(self):
        """Get the minor matrix of this matrix (minors of each element)"""
        return Matrix([[self.minor(i, j) for j in range(self.n_cols)] for i in range(self.n_rows)])

    def cofactor(self, i, j):
        """Get the cofactor of element M[i][j]"""
        return self.minor(i, j) * pow(-1, i + j)

    def cofactor_matrix(self):
        """Get the cofactor matrix of this matrix"""
        return Matrix([[self.cofactor(i, j) for j in range(self.n_cols)] for i in range(self.n_rows)])

    def adj(self):
        """Computes the adjoint of the matrix"""
        return self.cofactor_matrix().transpose()

    def det(self):
        """Recursively computes determinant of the matrix"""
        if self.n_cols != self.n_rows:
            raise NotImplementedError("Only possible for square matrices")
        if self.n_cols == 0:
            return None
        if self.n_cols == 1:
            return self[0][0]

        summ = self[0][0] * self.minor(0, 0)
        sign = -1
        for i in range(1, len(self[0])):
            summ += (self[0][i] * self.minor(0, i)) * sign
            sign *= -1

        return summ

    def __abs__(self):
        return self.det()

    def inv(self):
        """Compute the inverse of the matrix"""
        return self.adj() * (1 / self.det())


if __name__ == '__main__':
    # Some day I might add actual tests here

    v1 = Vector(3, 2)
    v2 = Vector(3, 1)

    m1 = Matrix([[1, -4, 2], [4, -7, 6], [1, 2, 3], [3, 4, 2]])
    m2 = Matrix.from_cols([[1, -4, 2], [4, -7, 6], [1, 2, 3], [3, 4, 2]])

