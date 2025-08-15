import requests
from web3 import Web3
import random
import time


RPC_URL = "https://testnet.dplabs-internal.com"
CHAIN_ID = 688688
API_URL = "https://api.pharosnetwork.xyz"
DELAY_BETWEEN_TX = 8 


w3 = Web3(Web3.HTTPProvider(RPC_URL))


def generate_random_address():
    return w3.eth.account.create().address

def sign_message(private_key, message="pharos"):
    from eth_account.messages import encode_defunct
    acct = w3.eth.account.from_key(private_key)
    msg = encode_defunct(text=message)
    signed = acct.sign_message(msg)
    return signed.signature.hex()

def login_with_private_key(private_key):
    address = w3.eth.account.from_key(private_key).address
    signature = sign_message(private_key)
    url = f"{API_URL}/user/login?address={address}&signature={signature}"
    headers = {
        "Origin": "https://testnet.pharosnetwork.xyz",
        "Referer": "https://testnet.pharosnetwork.xyz"
    }
    response = requests.post(url, headers=headers)
    try:
        data = response.json()
        token = data.get("data", {}).get("jwt")
        return token
    except:
        return None

def send_transaction(private_key, to_address, value=0.001):
    account = w3.eth.account.from_key(private_key)
    address = account.address
    value_wei = w3.to_wei(value, 'ether')
    nonce = w3.eth.get_transaction_count(address)

    tx = {
        'chainId': CHAIN_ID,
        'to': to_address,
        'value': value_wei,
        'gas': 21000,
        'gasPrice': w3.to_wei(1.2, 'gwei'),
        'nonce': nonce,
    }

    signed_tx = w3.eth.account.sign_transaction(tx, private_key)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
    w3.eth.wait_for_transaction_receipt(tx_hash)

    return tx_hash.hex(), address

def verify_transaction(address, tx_hash, bearer_token):
    url = f"{API_URL}/task/verify?address={address}&task_id=103&tx_hash={tx_hash}"
    headers = {
        'Authorization': f'Bearer {bearer_token}',
        'Origin': 'https://testnet.pharosnetwork.xyz',
        'Referer': 'https://testnet.pharosnetwork.xyz'
    }
    return requests.post(url, headers=headers).json()

def get_profile_info(address, bearer_token):
    url = f"{API_URL}/user/profile?address={address}"
    headers = {
        'Authorization': f'Bearer {bearer_token}',
        'Origin': 'https://testnet.pharosnetwork.xyz',
        'Referer': 'https://testnet.pharosnetwork.xyz'
    }
    return requests.get(url, headers=headers).json()


def main():
    try:
        jumlah_tx = int(input("Jumlah transaksi per wallet: "))
    except ValueError:
        print("Masukkan angka yang valid.")
        return

    try:
        with open("privateKeys.txt", "r") as f:
            private_keys = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("File privateKeys.txt tidak ditemukan.")
        return

    for i, pk in enumerate(private_keys):
        print("="*50)
        print(f"▶️ Wallet #{i+1}")

        if not pk.startswith("0x"):
            pk = "0x" + pk

        try:
            account = w3.eth.account.from_key(pk)
            address = account.address
            print(f"Wallet Address: {address}")
        except:
            print("❌ Private key tidak valid.")
            continue

        
        token = login_with_private_key(pk)
        if not token:
            print("❌ Login gagal.")
            continue
        print("✅ Login berhasil!")

        for txi in range(1, jumlah_tx + 1):
            print(f"\n🔄 Transaksi #{txi}")
            tujuan = wallet.txt()
            print(f"Alamat tujuan: {tujuan}")

            try:
                tx_hash, sender = send_transaction(pk, tujuan)
                print(f"✅ Transaksi berhasil: {tx_hash}")

                
                verif = verify_transaction(sender, tx_hash, token)
                if verif.get("code") == 0 and verif.get("data", {}).get("verified"):
                    print("🔒 Verifikasi sukses!")
                else:
                    print("❌ Verifikasi gagal.")

                profil = get_profile_info(sender, token)
                poin = profil.get("data", {}).get("user_info", {}).get("TaskPoints", 0)
                print(f"🎯 Total Poin: {poin}")

                if txi < jumlah_tx:
                    print(f"⏳ Menunggu {DELAY_BETWEEN_TX} detik...\n")
                    time.sleep(DELAY_BETWEEN_TX)

            except Exception as e:
                print(f"❗ Kesalahan: {str(e)}")
                print("⏸ Menunggu 5 detik...")
                time.sleep(5)

if __name__ == "__main__":
    main()
