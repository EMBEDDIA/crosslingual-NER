# Copyright (C) 2016-2018  Mikel Artetxe <artetxem@gmail.com>
#                    2019  Matej Ulčar <matej.ulcar@fri.uni-lj.si>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import extra.embeddings as embeddings
from extra.cupy_utils import *
import numpy as np

def vecmap_orth(x, w):
    embeddings.normalize(x, ['unit', 'center', 'unit'])
    xw = np.empty_like(x)
    x.dot(w, out=xw)
    return xw

def vecmap(x, W2, s): #src: W2=wx2, trg: W2=wz2
    #xp = get_cupy()
    xp = np
    xp.random.seed(0)
    # STEP 0: Normalize
    x = xp.asarray(x)
    s = xp.asarray(s)
    W2 = xp.asarray(W2)
    embeddings.normalize(x, ['unit', 'center', 'unit'])
    xw = xp.empty_like(x)
    xw[:] = x

    # STEP 1: Whitening
    # We do not apply whitening or de-whitening (step 4), due to errors
    # with these two operations during supervised mapping of ELMo embeddings

    def whitening_transformation(m):
        u, s, vt = xp.linalg.svd(m, full_matrices=False)
        return vt.T.dot(xp.diag(1/s)).dot(vt)

    #W1 = whitening_transformation(xw)
    #xw = xw.dot(W1)
    
    # STEP 2: Orthogonal mapping
    xw = xw.dot(W2)

    # STEP 3: Re-weighting
    xw *= s**0.5

    # STEP 4: De-whitening
    #xw = xw.dot(W2.T.dot(xp.linalg.inv(W1)).dot(W2))

    return xw

def batch_map(batch, W, s):
    tr_vec = []
    for v in batch:
        tr_vec.append(vecmap(np.array([v]), W, s))

    return np.array(tr_vec)

def batch_orth(batch, W):
    tr_vec = []
    for v in batch:
        tr_vec.append(vecmap_orth(np.array([v]), W))
    return np.array(tr_vec)

