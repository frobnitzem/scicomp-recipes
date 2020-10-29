How to contribute to this book
##############################

Submitting suggestions
====================================

Have a suggestion for improvement? Share it by
`opening an issue <https://code.ornl.gov/99R/olcf-cookbook/issues/new>`_


Authoring content
==================

Setup authoring environment
----------------------------

#. Install Sphinx locally:

   .. code-block:: bash

        $ pip3 install sphinx

   This can be in your home area, a virtual environment, container, etc.


#. Download a copy of the book source:

   .. code-block:: bash

    $ git clone https://code.ornl.gov/99R/olcf-cookbook

#. Build the book:

   .. code-block:: bash

    $ cd olcf-cookbook && make html

#. Locally preview the generated web pages

   Point your web browser to ``file:///<path to pwd>/_build/index.html``

Edit the book
-------------------------

#. Setup your environment as above
#. Create a new branch to make your edits:

   .. code-block:: bash

      $ git checkout -b my-edits-branch

#. Re-build and preview your edits:

   .. code-block:: bash

      $ make html

#. Commit your edits:

   .. code-block:: bash

      $ git commit -am "Describe your changes here."

#. Create a patchfile showing changes since the ``develop`` branch:

   .. code-block:: bash

      $ git format-patch develop

#. Email the authors directly and attach your patches.


Resources
---------------

| `Sphinx Quickstart <http://www.sphinx-doc.org/en/master/usage/quickstart.html>`_
| `restructuredText Primer <http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_
| `restructuredText Reference <http://docutils.sourceforge.net/rst.html>`_


GitHub Guidelines
===================

Here are some guidelines and common practices that we use in this project.

- When you want to work on an issue, assign it to yourself if no one is assigned
  yet. If there is somebody assigned, check in with that person about
  collaborating.

- Reference the issue(s) that your PR addresses with GitHub's '#' notation.

- Use "WIP" in your PR title to indicate that it should not be merged yet.
  Remove just the WIP when you are ready for it to be merged.

- You do not need to assign labels to your PR, but you may do so if you have
  suggestions. However, be aware that the labels might get changed.
