# ----------------------------------------------------------------------
# Compiling documentation
# ----------------------------------------------------------------------
FROM mambaorg/micromamba AS builder

USER root

# Set the GA_KEY environment variable
ARG GA_KEY
ENV GA_KEY=$GA_KEY

# install packages
WORKDIR /src
COPY environment.yml ./
ENV PATH "$MAMBA_ROOT_PREFIX/bin:$PATH"
RUN micromamba install -y -n base -c conda-forge -c in-core \
    sphinx=6.2.1 sphinx_rtd_theme -f environment.yml

# copy code and generate documentation
COPY . ./
RUN sphinx-build -v -b html docs/source docs/build

# Run the insert_ga_to_header.py script to insert Google Analytics code
RUN python /src/docs/source/insert_ga_to_header.py

# ----------------------------------------------------------------------
# Building actual container
# ----------------------------------------------------------------------
FROM nginx

COPY --from=builder /src/docs/build/ /usr/share/nginx/html/doc/pyincore_viz/
