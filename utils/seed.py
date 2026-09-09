def seed_everything(seed: int = 42) -> None: # {{{
    """Fix random seeds for reproducibility."""
    
    import random
    import os
    import numpy as np
    import torch
    import logging
    
    logger = logging.getLogger(__name__)
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    logger.info(f"Seed set to {seed}")
    
# }}}
