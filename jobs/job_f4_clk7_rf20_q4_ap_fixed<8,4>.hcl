job "scan_f4_clk7_rf20_q4_ap_fixed<8,4>" {
  datacenters = ["cerndc-olab"]
  type = "batch"
  group "hls4ml-scan" {
    count = 1
    task "synth_scan" {
      user = "hls4ml"
      driver = "docker"
      config {
        image = "gitlab-registry.cern.ch/fastmachinelearning/hls4ml-testing:0.3.vivado"
        privileged = true
        network_mode = "host"
        args = ["/bin/bash", "-c", "cd ../../local && git clone https://github.com/nicologhielmetti/enet-script && cd enet-script && chmod +x run_enet_explore.sh && ./run_enet_explore.sh -r20 -f4 -c7 -q4 -p'ap_fixed<8,4>' -iX_test.npy -oy_test.npy"]
      }
      resources {
        memory = 32000
      }
    }
  }
}