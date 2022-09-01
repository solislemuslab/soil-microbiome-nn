FROM ubuntu:20.04
RUN apt-get update
RUN apt-get install -y build-essential
RUN apt-get install -y wget
RUN wget https://www.cs.toronto.edu/~radford/ftp/fbm.2022-04-21.tar
RUN tar xf fbm.2022-04-21.tar
RUN mv fbm.2022-04-21 software
WORKDIR software
RUN ./make-all
WORKDIR ..
ENV PATH="${PATH}:/software/bin-fbm"
RUN rm fbm.2022-04-21.tar
