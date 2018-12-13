#! /bin/bash
if [ -z "$1" ]; then
  echo "Please provide a username, e.g. gs15687"
  exit 1
else
  echo "Setting up for $1"
fi

if grep -q snowy ~/.ssh/config; then
  echo "Snowy configuration present"
else
  echo "No snowy configuration present -> Configuring"
  echo "Host snowy" >> ~/.ssh/config
  echo "    HostName snowy.cs.bris.ac.uk" >> ~/.ssh/config
  echo "    User $1" >> ~/.ssh/config
fi

if grep -q bcp4 ~/.ssh/config; then
  echo "Bluecrystal phase 4 configuration present"
else
  echo "No bluecrystal phase 4 configuration present -> Configuring"
  echo "Host bcp4" >> ~/.ssh/config
  echo "    HostName bc4login.acrc.bris.ac.uk" >> ~/.ssh/config
  echo "    User $1" >> ~/.ssh/config
  echo "    ProxyCommand ssh snowy nc %h %p 2> /dev/null" >> ~/.ssh/config
fi

if [[ "$OSTYPE" == "linux-gnu" ]]; then
  echo "Done"
elif [[ "$OSTYPE" == "darwin"* ]]; then
  brew update
  brew install watchman
  echo "Done"
else
  echo "Cannot install watchman"
fi