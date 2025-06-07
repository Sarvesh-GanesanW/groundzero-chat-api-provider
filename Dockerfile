FROM public.ecr.aws/lambda/python:3.9.2024.10.16.12 AS builder
ENV ODBCINI=/opt/odbc.ini
ENV ODBCSYSINI=/opt/
ARG UNIXODBC_VERSION=2.3.9

RUN yum install -y gzip tar openssl-devel && yum groupinstall "Development Tools" -y

RUN curl ftp://ftp.unixodbc.org/pub/unixODBC/unixODBC-${UNIXODBC_VERSION}.tar.gz -O \
    && tar xzvf unixODBC-${UNIXODBC_VERSION}.tar.gz \
    && cd unixODBC-${UNIXODBC_VERSION} \
    && ./configure --sysconfdir=/opt --disable-gui --disable-drivers --enable-iconv --with-iconv-char-enc=UTF8 --with-iconv-ucode-enc=UTF16LE --prefix=/opt \
    && make \
    && make install

RUN curl https://packages.microsoft.com/config/rhel/6/prod.repo > /etc/yum.repos.d/mssql-release.repo
RUN yum install e2fsprogs.x86_64 0:1.43.5-2.43.amzn1 fuse-libs.x86_64 0:2.9.4-1.18.amzn1 libss.x86_64 0:1.43.5-2.43.amzn1 -y
RUN ACCEPT_EULA=Y yum install -y msodbcsql17

ENV CFLAGS="-I/opt/include"
ENV LDFLAGS="-L/opt/lib"

RUN mkdir /opt/python/ && cd /opt/python/ && pip install fastapi uvicorn[standard] boto3 pydantic PyPDF2 pandas openpyxl python-multipart pgvector sqlalchemy pyjwt awsgi aws-wsgi mangum
COPY requirements.txt /opt/python
RUN cd /opt/python

COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install -r requirements.txt
COPY . ${LAMBDA_TASK_ROOT}

CMD [ "index.handler" ]