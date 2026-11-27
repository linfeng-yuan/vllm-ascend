ARG BASE=local/r148:required
FROM ${BASE}
ADD runtime.tar.gz /
ADD extras.tar.gz /
COPY check-runtime.sh /usr/local/bin/dsv41-check-runtime
RUN bash /usr/local/bin/dsv41-check-runtime
LABEL org.opencontainers.image.revision="60ece1a4bc09480e014a28a58e6ca4ea7bd22abd"
