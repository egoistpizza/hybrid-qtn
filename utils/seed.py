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
    
    logger.debug(f"\x1b[38;5;240mrandom.seed({seed})\x1b[0m")
    logger.debug(f"\x1b[38;5;240mos.environ['PYTHONHASHSEED'] = str({seed})\x1b[0m")
    logger.debug(f"\x1b[38;5;240mnp.random.seed({seed})\x1b[0m")
    logger.debug(f"\x1b[38;5;240mtorch.manual_seed({seed})\x1b[0m")
    logger.debug(f"\x1b[38;5;240mtorch.cuda.manual_seed({seed})\x1b[0m")
    logger.debug(f"\x1b[38;5;240mtorch.cuda.manual_seed_all({seed})\x1b[0m")
    logger.debug(f"\x1b[38;5;240mtorch.backends.cudnn.deterministic = True\x1b[0m")
    logger.debug(f"\x1b[38;5;240mtorch.backends.cudnn.benchmark = True\x1b[0m")
# }}}