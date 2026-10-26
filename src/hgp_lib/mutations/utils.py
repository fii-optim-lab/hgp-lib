class MutationError(Exception):
    # Placeholder class used to signify that a mutation could not be executed because it would violate constraints
    # TODO: Write in documentation that when creating mutations, mutations should not leave a rule in an invalid state.
    # If they do, they should revert the changes and just raise MutationError
    pass
