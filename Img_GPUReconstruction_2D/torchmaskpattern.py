"""
@Title: URA/MURA Coded Mask Pattern Simulation with PyTorch
@Author: Edoardo Giancarli
@Date: 19/12/24
@Content:
    - URAMaskPattern: Generates a 2D URA pattern for a coded mask camera.
    - MURAMaskPattern: Generates a 2D MURA pattern for a coded mask camera.
@References:
    [1] codeapertures package, https://github.com/bpops
    [2] E. E. Fenimore and T. M. Cannon, "Coded aperture imaging with uniformly redundant arrays," Appl. Opt. 17, 337-347 (1978)
    [3] E. E. Fenimore and S. R. Gottesman, "New family of binary arrays for coded aperture imaging" Appl. Opt. 28 (20): 4344-4352 (1989)
"""

import torch
from sympy import isprime, primerange


class URAMaskPattern:
    """Generates a 2D URA pattern for a coded mask camera."""
    
    def __init__(self, rank: int):
        
        self.pattern_type = 'URA'

        self._check_rank(rank)
        self.rank = rank

        self.prime_pair = self._get_prime_pair(rank)
        C_r_i, C_s_j = self._get_pattern_root()
        self.basic_pattern = self._get_basic_pattern(C_r_i, C_s_j)
        self.basic_decoder = self._get_decoder()
    

    def _check_rank(self, rank):
        if rank < 0:
            raise ValueError(f"rank must be >= 0, got rank = {rank} instead.")
    

    def _get_prime_pair(self, rank) -> tuple[int, int]:

        assert rank >= 0

        lim = int(1e4)
        primes = list(primerange(2, lim))
        p1, this_rank = primes[0], -1

        for p2 in primes[1:]:
            if (p2 - p1) == 2:
                this_rank += 1
                if this_rank == rank:
                    return p2, p1
            p1 = p2

        raise ValueError(f"Could not find prime pairs in the range [2, {lim}] for rank = {rank}.")
    

    def _get_pattern_root(self) -> tuple[torch.tensor, torch.tensor]:

        r, s = self.prime_pair
        assert isprime(r)
        assert isprime(s)
        assert r - s == 2

        C_r_i = torch.zeros(r) - 1
        C_s_j = torch.zeros(s) - 1

        for x in range(1, r):
            C_r_i[x**2 % r] = 1
        for y in range(1, s):
            C_s_j[y**2 % s] = 1
        
        return C_r_i, C_s_j


    def _get_basic_pattern(self, C_r_i, C_s_j) -> torch.tensor:

        A = torch.zeros(self.prime_pair)

        for i in range(self.prime_pair[0]):
            for j in range(self.prime_pair[1]):

                if i == 0: A[i,j] = 0
                elif j == 0: A[i,j] = 1
                elif C_r_i[i]*C_s_j[j] == 1: A[i,j] = 1
                else: A[i,j] = 0
        
        return A
    

    def _get_decoder(self) -> torch.tensor:

        G = 2*self.basic_pattern - 1
        G /= self.basic_pattern.sum()

        return G




class MURAMaskPattern:
    """Generates a 2D MURA pattern for a coded mask camera."""
    
    def __init__(self, rank: int):
        
        self.pattern_type = 'MURA'

        self._check_rank(rank)
        self.rank = rank

        self.l = self._get_prime(rank)
        C_r_i, C_s_j = self._get_pattern_root()
        self.basic_pattern = self._get_basic_pattern(C_r_i, C_s_j)
        self.basic_decoder = self._get_decoder()
    

    def _check_rank(self, rank):
        if rank < 0:
            raise ValueError(f"rank must be >= 0, got rank = {rank} instead.")
    

    def _get_prime(self, rank) -> int:

        assert rank >= 0
        m, this_rank = 1, -1

        while True:
            l = 4*m + 1
            if isprime(l):
                this_rank += 1
                if this_rank == rank:
                    return l
            m += 1
    

    def _get_pattern_root(self) -> tuple[torch.tensor, torch.tensor]:
        
        assert isprime(self.l)

        C_r_i = torch.zeros(self.l) - 1

        for x in range(1, self.l):
            C_r_i[x**2 % self.l] = 1
        
        C_s_j = C_r_i.clone()
        
        return C_r_i, C_s_j
    

    def _get_basic_pattern(self, C_r_i, C_s_j) -> torch.tensor:

        A = torch.zeros((self.l, self.l))

        for i in range(self.l):
            for j in range(self.l):

                if i == 0: A[i,j] = 0
                elif j == 0: A[i,j] = 1
                elif C_r_i[i]*C_s_j[j] == 1: A[i,j] = 1
        
        return torch.t(A)
    

    def _get_decoder(self) -> torch.tensor:

        G = 2*self.basic_pattern - 1
        G[0, 0] = 1
        G /= self.basic_pattern.sum()

        return G


# end