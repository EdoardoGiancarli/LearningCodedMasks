"""
Temporary module for new IROS reconstruction procedure.
In this version, the sub-logics are made flexible by allowing for customisation.
The IROS routine is now intended as a "wrapper" for the main logics (source finding process, parameters fitting and source subtraction).
"""

# search for updated versions of:
#   - finder, fitter, subtractor methods
#   - optimiser method
#
# NOTE: inside the finder there is the sky pos masking
# NOTE: the optimiser is called inside the fitter
#
# NOTE (optimiser): custom obj for `curve_fit` output for `verbose` func input
# NOTE (optimiser): general custom obj for optimising procedure? Some scipy routine have their own output obj...
# NOTE (optimiser): make `verbose` func flexible by again giving it as input


# end