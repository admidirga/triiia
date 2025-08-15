require('dotenv').config();
const axios = require('axios');
const fs = require('fs');
const { ethers } = require('ethers');
const { HttpsProxyAgent } = require('https-proxy-agent');

// Load wallets & proxies
const wallets = JSON.parse(fs.readFileSync('wallets.json'));
const proxies = fs.readFileSync('proxy.txt', 'utf-8').split('\n').filter(Boolean);

// ENV
const rpc = process.env.RPC_URL;
const provider = new ethers.providers.JsonRpcProvider(rpc);

// ABIs
const ABI_ERC20 = [
  "function approve(address spender, uint256 amount) public returns (bool)",
  "function balanceOf(address) view returns (uint256)"
];
const ABI_MINT = ["function mintAUSD(uint256 amount) public"];
const ABI_STAKE = ["function stake(uint256 _tokens) public"];

// Addresses
const TOKENS = {
  ATH: "0x1428444Eacdc0Fd115dd4318FcE65B61Cd1ef399",
  USD1: "0x16a8A3624465224198d216b33E825BcC3B80abf7",
  USDe: "0xf4BE938070f59764C85fAcE374F92A4670ff3877",
  LVLUSD: "0x8802b7bcF8EedCc9E1bA6C20E139bEe89dd98E83",
  AI16Z: "0x2d5a4f5634041f50180A25F26b2A8364452E3152",
  VIRTUAL: "0xFF27D611ab162d7827bbbA59F140C1E7aE56e95C",
  VANA: "0xBEbF4E25652e7F23CCdCCcaaCB32004501c4BfF8",
};
const CONTRACTS = {
  mintAUSD: "0x2cFDeE1d5f04dD235AEA47E1aD2fB66e3A61C13e",
  stakeAUSD: "0x054de909723ECda2d119E31583D40a52a332f85c",
  stakeUSD1: "0x7799841734Ac448b8634F1c1d7522Bc8887A7bB9",
  stakeUSDe: "0x3988053b7c748023a1aE19a8ED4c1Bf217932bDB",
  stakeLVLUSD: "0x5De3fBd40D4c3892914c3b67b5B529D776A1483A",
  mintAZUSD: "0xB0b53d8B4ef06F9Bbe5db624113C6A5D35bB7522",
  stakeAI16Z: "0xf45Fde3F484C44CC35Bdc2A7fCA3DDDe0C8f252E",
  mintVIRTUAL: "0x3dCACa90A714498624067948C092Dd0373f08265",
  stakeVIRTUAL: "0x5bb9Fa02a3DCCDB4E9099b48e8Ba5841D2e59d51",
  mintVANA: "0xEfbAE3A68b17a61f21C7809Edfa8Aa3CA7B2546f",
  stakeVANA: "0x2608A88219BFB34519f635Dd9Ca2Ae971539ca60"
};

// Faucet endpoints
const faucetUrls = [
  "https://app.x-network.io/maitrix-faucet/faucet",
  "https://app.x-network.io/maitrix-usde/faucet",
  "https://app.x-network.io/maitrix-lvl/faucet",
  "https://app.x-network.io/maitrix-virtual/faucet",
  "https://app.x-network.io/maitrix-vana/faucet",
  "https://app.x-network.io/maitrix-ai16z/faucet",
  "https://app.x-network.io/maitrix-usd1/faucet"
];

// Headers
const headers = {
  "Content-Type": "application/json",
  "Origin": "https://app.testnet.themaitrix.ai",
  "Referer": "https://app.testnet.themaitrix.ai/",
  "User-Agent": "Mozilla/5.0"
};

// Delay helper
const delay = (ms) => new Promise(res => setTimeout(res, ms));

// Faucet claim
async function claimAllFaucets(address) {
  console.log(`💧 Claiming faucets for ${address}`);
  for (let url of faucetUrls) {
    const proxy = proxies[Math.floor(Math.random() * proxies.length)];
    const agent = proxy ? new HttpsProxyAgent(proxy) : null;

    try {
      const res = await axios.post(url, { address }, {
        headers,
        httpsAgent: agent,
        timeout: 10000
      });
      console.log(`✅ Faucet OK: ${url}`, res.data);
    } catch (err) {
      console.error(`❌ Faucet Fail ${url}:`, err.response?.data || err.message);
    }
  }
}

// Mint AUSD
async function mintAUSD(wallet) {
  const amount = ethers.utils.parseUnits("50", 18);
  const token = new ethers.Contract(TOKENS.ATH, ABI_ERC20, wallet);
  const contract = new ethers.Contract(CONTRACTS.mintAUSD, ABI_MINT, wallet);

  try {
    const balance = await token.balanceOf(wallet.address);
    console.log(`🔎 ATH Balance: ${ethers.utils.formatUnits(balance, 18)} ATH`);

    if (balance.lt(amount)) {
      console.log("❌ ATH balance kurang untuk mint");
      return;
    }

    const approveTx = await token.approve(CONTRACTS.mintAUSD, amount);
    await approveTx.wait();
    console.log(`✅ Approved ATH: ${approveTx.hash}`);

    const mintTx = await contract.mintAUSD(amount);
    await mintTx.wait();
    console.log(`✅ Minted AUSD: ${mintTx.hash}`);
  } catch (err) {
    console.error("❌ Mint Error:", err.reason || err.message);
  }
}

// Stake function
async function stakeToken(wallet, tokenAddress, contractAddress, tokenName) {
  const token = new ethers.Contract(tokenAddress, ABI_ERC20, wallet);
  const contract = new ethers.Contract(contractAddress, ABI_STAKE, wallet);

  try {
    const balance = await token.balanceOf(wallet.address);
    if (balance.isZero()) {
      console.log(`❌ ${tokenName} balance = 0, skip`);
      return;
    }

    const approveTx = await token.approve(contractAddress, balance);
    await approveTx.wait();
    console.log(`✅ Approved ${tokenName}: ${approveTx.hash}`);

    const stakeTx = await contract.stake(balance);
    await stakeTx.wait();
    console.log(`✅ Staked ${tokenName}: ${stakeTx.hash}`);
  } catch (err) {
    console.error(`❌ ${tokenName} Stake Error:`, err.reason || err.message);
  }
}

// Per wallet
async function processWallet({ address, privateKey }) {
  console.log(`\n🚀 Processing wallet: ${address}`);
  const wallet = new ethers.Wallet(privateKey, provider);
  await claimAllFaucets(address);
  await mintAUSD(wallet);
  await stakeToken(wallet, CONTRACTS.stakeAUSD, CONTRACTS.stakeAUSD, "AUSD");
  await stakeToken(wallet, TOKENS.USDe, CONTRACTS.stakeUSDe, "USDe");
  await stakeToken(wallet, TOKENS.LVLUSD, CONTRACTS.stakeLVLUSD, "LVLUSD");
  console.log(`✅ Done for ${address}`);
}

// Main loop
async function main() {
  while (true) {
    for (const [i, w] of wallets.entries()) {
      await processWallet(w);
      if (i < wallets.length - 1) {
        console.log("⏳ Wait 10 seconds before next wallet...");
        await delay(10000);
      }
    }

    console.log("🕛 Sesi selesai. Tunggu 24 jam sebelum mulai ulang...\n");
    for (let sisa = 24 * 60 * 60; sisa > 0; sisa--) {
      process.stdout.write(`⏳ Ulang dalam ${sisa} detik...\r`);
      await delay(1000);
    }
  }
}

main();
