FROM python:3.9.9

ARG WORKDIR=/pymosaic
WORKDIR ${WORKDIR}

# for apt install to avoid errors caused by launching dialog box
ENV DEBIAN_FRONTEND=noninteractive

# to avoid hang-up in docker build
ENV TZ=Asia/Tokyo
RUN echo $TZ > /etc/timezone

# change default shell (/bin/sh -> /bin/bash)
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN chsh -s /bin/bash

# Increase timeout for apt-get to 300 seconds
# ref: https://yamap55.hatenablog.com/entry/2020/02/27/212922
RUN /bin/echo -e "\n\
    Acquire::http::Timeout \"300\";\n\
    Acquire::ftp::Timeout \"300\";" >> /etc/apt/apt.conf.d/99timeout

# Configure apt and install packages
RUN rm -rf /var/lib/apt/lists/* \
    && apt-get update \
    && apt-get upgrade -y \
    && apt-get -y --no-install-recommends install sudo vim tzdata less \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# create non-root-user & sudo support (default: name -> developer, user/group ID -> 1000)
# ref: https://aka.ms/vscode-remote/containers/non-root-user.
ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# terminal setting (git-completion & git-prompt)
RUN wget --progress=dot:giga https://raw.githubusercontent.com/git/git/master/contrib/completion/git-completion.bash -O /home/developer/.git-completion.bash \
    && wget --progress=dot:giga https://raw.githubusercontent.com/git/git/master/contrib/completion/git-prompt.sh -O /home/developer/.git-prompt.sh \
    && chmod a+x /home/developer/.git-completion.bash \
    && chmod a+x /home/developer/.git-prompt.sh \
    && echo -e "\n\
    source ~/.git-completion.bash\n\
    source ~/.git-prompt.sh\n\
    export PS1='\\[\\e]0;\\u@\\h: \\w\\a\\]\${debian_chroot:+(\$debian_chroot)}\\[\\033[01;32m\\]\\u\\[\\033[00m\\]:\\[\\033[01;34m\\]\\w\\[\\033[00m\\]\\[\\033[1;30m\\]\$(__git_ps1)\\[\\033[0m\\] \\$ '\n\
    " >>  /home/developer/.bashrc


# pip install
RUN pip install --no-cache-dir -U pip

# restore setting
ENV DEBIAN_FRONTEND=

# shell start
CMD ["/bin/bash"]
