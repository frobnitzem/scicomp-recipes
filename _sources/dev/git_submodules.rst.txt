Git Submodules
##############

Create a new project and add in slate as a git submodule:

.. code-block:: bash

    mkdir awesomesauce
    cd awesomesauce
    git init
    git commit -am "Empty repo."

    git submodule add https://bitbucket.org/icl/slate/
    git commit -am "Added slate as a submodule."

    cd ../
    git clone awesomesauce anothersauce
    git submodule init
    git submodule update

I'd use submodules for a use-case where the dependency project
doesn't install to the system yet, but is relatively stable
and stands well on its own.

[Further reading](https://git-scm.com/book/en/v2/Git-Tools-Submodules)
