Useful Shortcuts for Makefiles
##############################

When starting out new projects, Makefiles
can be faster and easier to configure than cmake.

Makefiles take an opposing world-view to CMake.
Where CMake tries to auto-detect install and configure
options, make works best for a static build tree
with fixed dependencies configured explicitly
by the user.

Not second-guessing, autodetecting or otherwise overriding
the user's environment is make's speciality.

There are many `good guides online <https://nullprogram.com/blog/2017/08/20/>_`.

Syntax Example
--------------

Here's an example of good practices from the `quark project <https://tools.suckless.org/quark/>_`:

.. code-block:: make

    # See LICENSE file for copyright and license details
    # quark - simple web server
    .POSIX:

    include config.mk

    COMPONENTS = data http sock util

    all: quark

    data.o: data.c data.h http.h util.h config.mk
    http.o: http.c config.h http.h util.h config.mk
    main.o: main.c arg.h data.h http.h sock.h util.h config.mk
    sock.o: sock.c sock.h util.h config.mk
    util.o: util.c util.h config.mk

    quark: config.h $(COMPONENTS:=.o) $(COMPONENTS:=.h) main.o config.mk
            $(CC) -o $@ $(CPPFLAGS) $(CFLAGS) $(COMPONENTS:=.o) main.o $(LDFLAGS)

    config.h:
            cp config.def.h $@

    clean:
            rm -f quark main.o $(COMPONENTS:=.o)

    dist:
            rm -rf "quark-$(VERSION)"
            mkdir -p "quark-$(VERSION)"
            cp -R LICENSE Makefile arg.h config.def.h config.mk quark.1 \
                    $(COMPONENTS:=.c) $(COMPONENTS:=.h) main.c "quark-$(VERSION)"
            tar -cf - "quark-$(VERSION)" | gzip -c > "quark-$(VERSION).tar.gz"
            rm -rf "quark-$(VERSION)"

    install: all
            mkdir -p "$(DESTDIR)$(PREFIX)/bin"
            cp -f quark "$(DESTDIR)$(PREFIX)/bin"
            chmod 755 "$(DESTDIR)$(PREFIX)/bin/quark"
            mkdir -p "$(DESTDIR)$(MANPREFIX)/man1"
            cp quark.1 "$(DESTDIR)$(MANPREFIX)/man1/quark.1"
            chmod 644 "$(DESTDIR)$(MANPREFIX)/man1/quark.1"

    uninstall:
            rm -f "$(DESTDIR)$(PREFIX)/bin/quark"
            rm -f "$(DESTDIR)$(MANPREFIX)/man1/quark.1"

Notice that explicit dependencies ensure the project
is properly rebuilt when relevant files are edited.

Also, special make substituions are used:
* ``$@`` = the name of the rule's target (e.g. the ``config.h`` rule)
* ``$^`` = all dependencies of the rule
* ``$(VAR:a=b) = substitute trailing ``a`` with ``b`` in each word.

General Rules
-------------

The example above doesn't specifically say how to
compile ``.c`` files to ``.o`` files.  Instead,
it relies on the default rule.

We reproduce this below, along with some other useful
ones for modern software:

.. code-block:: make

    .SUFFIXES: c cx f cu

    .c.o:
            $(CC) $(CFLAGS) -c $<

    .f.o:
            $(FC) $(FFLAGS) -c $<

    .cc.o:
            $(CXX) $(CXXFLAGS) -c $<

    .cu.o:
            $(NVCC) $(NVCCFLAGS) -c $<

External Packages
-----------------

The quark example above uses a small, separate ``config.mk`` file
to let the user point the build at all system dependencies:

.. code-block:: make

    # quark version
    VERSION = 0

    # Customize below to fit your system

    # paths
    PREFIX = /usr/local
    MANPREFIX = $(PREFIX)/share/man

    # flags
    CPPFLAGS = -DVERSION=\"$(VERSION)\" -D_DEFAULT_SOURCE -D_XOPEN_SOURCE=700 -D_BSD_SOURCE
    CFLAGS   = -std=c99 -pedantic -Wall -Wextra -Os
    LDFLAGS  = -s

    # compiler and linker
    CC = cc

GNU Make defines the `shell function <https://www.gnu.org/software/make/manual/html_node/Shell-Function.html#Shell-Function>_`.
This can be helpful for adding limited auto-detection to ``config.mk``.
For example, ``$(shell pkg-config --libs zlib)``.

Interesting Tricks
------------------

GCC can create Makefiles listing user header files with ``-MMD``.
The way to use this is `like this <https://stackoverflow.com/questions/11855386/using-g-with-mmd-in-makefile-to-automatically-generate-dependencies>_`:

.. code-block:: make

    CXX = g++
    CXXFLAGS = -g -Wall -MMD      # The MMD flag causes x.d, etc. to be output
    OBJECTS = x.o y.o z.o         # object files forming executable
    DEPENDS = ${OBJECTS:.o=.d}    # substitutes ".o" with ".d"
    EXEC = a.out                  # executable name

    ${EXEC} : ${OBJECTS}          # link step
        ${CXX} ${OBJECTS} -o ${EXEC}

    -include ${DEPENDS}           # include x.d, y.d, and z.d

The ``-`` prefix on include prevents an error from being thrown
if the include command fails.

