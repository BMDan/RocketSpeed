package org.rocketspeed;

import java.util.ArrayList;

public class Configuration {

  private final ConfigurationImpl impl;

  public Configuration() {
    impl = new ConfigurationImpl(new ArrayList<HostId>(), new ArrayList<HostId>());
  }

  public void addPilot(String hostName, int port) {
    impl.getPilots().add(new HostId(hostName, port));
  }

  public void addCopilot(String hostName, int port) {
    impl.getCopilots().add(new HostId(hostName, port));
  }

  /* package */ ConfigurationImpl djinni() {
    return impl;
  }
}
