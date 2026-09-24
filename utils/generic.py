import argparse

""" Other utility function and variables shared across the project """


def argparse_str_to_bool(boolstr_value): # {{{
    """Argparse helper to infer boolean from string"""
    val = __internal_argparse_str_to_bool__(boolstr_value)
    if val is not None:
        return val
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected ({__true_strs__+__false_strs__})')
# }}}

__true_strs__ = ('yes', 'true', 't', 'y', '1')
__false_strs__ = ('no', 'false', 'f', 'n', '0')

def __internal_argparse_str_to_bool__(boolstr_value): # {{{
    if isinstance(boolstr_value, bool):
        return boolstr_value
    if boolstr_value.lower() in __true_strs__:
        return True
    elif boolstr_value.lower() in __false_strs__:
        return False
    else:
        return None
# }}}

def argparse_str_to_bool_fn_extra(list_extra_choices): # {{{
    """Function that returns a modified function for argparse helper to infer boolean from string, that also accepts the given strings"""
    def argparse_custom_funct(boolstr_or_regular_str): # {{{
        val = __internal_argparse_str_to_bool__(boolstr_or_regular_str)
        
        # A boolstring
        if val is not None:
            return val
        
        # A non-boolstring but one of the expected other values
        elif boolstr_or_regular_str.lower() in [choice.lower() for choice in list_extra_choices]:
            return boolstr_or_regular_str
        
        # Any other string
        else:
            raise argparse.ArgumentTypeError(f'Boolean value ({__true_strs__+__false_strs__}) or any of the {list_extra_choices} expected')
    # }}}
    
    return argparse_custom_funct
# }}}
