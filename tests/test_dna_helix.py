import pytest
import torch
from dna_helix import DNAHELIX, create_dna_helix

class TestDNAHELIX:
    def test_init(self):
        m = DNAHELIX(dim=256, hidden_dim=512)
        assert m.dim == 256
    
    def test_forward(self):
        m = DNAHELIX(dim=256)
        x = torch.randn(32, 256)
        assert m(x).shape == (32, 256)
    
    def test_factory(self):
        m = create_dna_helix()
        assert isinstance(m, DNAHELIX)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
