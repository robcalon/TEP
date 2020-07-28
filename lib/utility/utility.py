import os

def save_figure(fig, name, dir, dpi=300):
    """
    Convenient function to save figures
    
    Parameters
    ----------
    fig : Figure
        Figure that is saved.
    name : str
        Name of the saved figure.
    dir : str
        Path to save directory.
    dpi : int
        Dots per inch of saved figure.
    """

    if not os.path.exists(dir):
        os.makedirs(dir)

    fig.savefig('{}/{}_{}dpi.png'.format(dir, name, dpi), dpi=dpi,
                bbox_inches='tight', format='png')

    
def invert_dict(dic, lst=False):
    """
    Helper function to invert the key-value pairs in a dictonairy.
    
    Parameters
    ----------
    dic : dict
        Dictonairy to invert.
    lst : bool, default False
        Input dictonairy values contain lists.
        
    Return
    ------
    cid : dict
        Returns an inverted dictionairy.
    """
    
    # create the inverted dict
    cid = {}

    # iterate over each key-value pair
    for key, value in dic.items(): 

        # check if value is already a key
        if value not in cid: 
            
            if lst:
                # invert key to value as list item
                cid[value] = [key]
            else:
                # invert key to value
                cid[value] = key
        else: 
            # append key-value pair as value-key.
            cid[value].append(key) 
            
    return cid