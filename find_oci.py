import oci

config = oci.config.from_file()
compute_client = oci.core.ComputeClient(config)
network_client = oci.core.VirtualNetworkClient(config)

instance_id = "ocid1.instance.oc1.uk-london-1.anwgiljtzhomqaqc2jvl7gmmujkpzein3e4wuq3osozjkor6mgb3vzrgcieq"

# Get instance details
instance = compute_client.get_instance(instance_id).data

# Get attached VNICs (network interfaces)
vnic_attachments = compute_client.list_vnic_attachments(
    compartment_id=instance.compartment_id,
    instance_id=instance_id
).data

if vnic_attachments:
    vnic_id = vnic_attachments[0].vnic_id
    vnic = network_client.get_vnic(vnic_id).data
    print(f"Public IP: {vnic.public_ip}")
    print(f"Private IP: {vnic.private_ip}")
else:
    print("No network interface found")