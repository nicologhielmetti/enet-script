job "scan_f{n_filters}_clk{clock_period}_rf{reuse_factor}_q{quantization}_{precision}" {{
  datacenters = ["cerndc-olab"]
  type = "batch"
  group "hls4ml-scan" {{
    count = 1
    task "synth_scan" {{
      user = "hls4ml"
      driver = "docker"
      artifact {{
        source = "git::https://github.com/nicologhielmetti/enet-script"
        destination = "local/enet-scan"
      }}
      config {{
        image = "gitlab-registry.cern.ch/fastmachinelearning/hls4ml-testing:0.3.vivado"
        privileged = true
        args = ["/bin/bash", "local/enet-scan/run_enet_explore.sh {reuse_factor} {n_filters} {clock_period} {quantization} {precision} {input_data} {output_predictions} > local/output.txt"]
      }}
      resources {{
        memory = 8000
      }}
    }}
  }}
}}