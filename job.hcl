job "scan_f{n_filters}_clk{clock_period}_rf{reuse_factor}_q{quantization}_{precision}" {{
  datacenters = ["cerndc-olab"]
  type = "batch"
  group "hls4ml-scan" {{
    count = 1
    task "synth_scan" {{
      user = "hls4ml"
      driver = "docker"
      config {{
        image = "gitlab-registry.cern.ch/fastmachinelearning/hls4ml-testing:0.3.vivado"
        privileged = true
        network_mode = "host"
        args = ["/bin/bash", "-c", "cd ../../local && git clone https://github.com/nicologhielmetti/enet-script && cd enet-script && chmod +x run_enet_explore.sh && ./run_enet_explore.sh -r{reuse_factor} -f{n_filters} -c{clock_period} -q{quantization} -p'{precision}' -i{input_data} -o{output_predictions}"]
      }}
      resources {{
        memory = 32000
      }}
    }}
  }}
}}