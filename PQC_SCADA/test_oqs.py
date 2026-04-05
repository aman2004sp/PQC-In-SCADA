import oqs

kemalg = "ML-KEM-512"

with oqs.KeyEncapsulation(kemalg) as kem:
    public_key = kem.generate_keypair()
    ciphertext, ss_enc = kem.encap_secret(public_key)
    ss_dec = kem.decap_secret(ciphertext)

    print("Algorithm:", kemalg)
    print("Shared secret match:", ss_enc == ss_dec)
