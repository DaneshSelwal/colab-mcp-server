import asyncio
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

WORKSPACE = Path(__file__).resolve().parent

PIPELINE_CODE = """
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("/content/titanic.csv")
df = df[["Survived", "Age", "Fare"]].dropna()

X = df[["Age", "Fare"]]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

print(f"Accuracy: {accuracy:.4f}")
""".strip()


def _extract_result_payload(result: Any) -> Any:
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return structured

    content = getattr(result, "content", None) or []
    texts: list[str] = []
    for item in content:
        text = getattr(item, "text", None)
        if text is not None:
            texts.append(text)

    if not texts:
        return None
    if len(texts) == 1:
        try:
            return json.loads(texts[0])
        except json.JSONDecodeError:
            return texts[0]
    return "\n".join(texts)


async def _call_tool(
    session: ClientSession,
    name: str,
    arguments: dict[str, Any],
    timeout_seconds: int = 600,
):
    result = await session.call_tool(
        name,
        arguments=arguments,
        read_timeout_seconds=timedelta(seconds=timeout_seconds),
    )
    if getattr(result, "isError", False):
        raise RuntimeError(f"Tool {name} failed: {_extract_result_payload(result)}")
    return _extract_result_payload(result)


async def main():
    server = StdioServerParameters(
        command="py",
        args=["-m", "uv", "run", "colab-mcp"],
        cwd=WORKSPACE,
    )

    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            connect_status = await _call_tool(
                session, "connect_colab", {}, timeout_seconds=10
            )

            print("\n>>> SERVER STARTED! BROWSER HANDSHAKE REQUIRED. <<<")
            print("Your Colab tab will NOT auto-discover the proxy.")
            print("Open your active Colab notebook URL and append this fragment, or paste the full URL below:")
            print(f"  Proxy port : {connect_status.get('proxy_port')}")
            print(f"  Proxy token: {connect_status.get('proxy_token')}")
            print(f"  Connect URL: {connect_status.get('connect_url')}")
            print("Then press Enter in the browser or refresh that exact URL.")
            print("Waiting 15 seconds for browser to connect...")

            connected = False
            for i in range(15):
                connect_status = await _call_tool(
                    session, "connect_colab", {}, timeout_seconds=10
                )
                if connect_status.get("connected") or connect_status.get(
                    "proxy_connected"
                ):
                    connected = True
                    print("\nSuccess! Browser connected to proxy.")
                    break
                await asyncio.sleep(1)
                print(".", end="", flush=True)

            if not connected:
                raise RuntimeError(
                    "\nColab tab never connected! Did you refresh the browser?"
                )

            print("\nRunning setup_ml_workspace...")
            await _call_tool(
                session,
                "setup_ml_workspace",
                {"packages": ["scikit-learn", "pandas"]},
            )

            print("Running fetch_remote_dataset...")
            await _call_tool(
                session,
                "fetch_remote_dataset",
                {
                    "download_url": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
                    "extract_to": "/content",
                },
            )

            print("Running execute_ml_pipeline...")
            pipeline_result = await _call_tool(
                session,
                "execute_ml_pipeline",
                {"code_block": PIPELINE_CODE},
                timeout_seconds=1200,
            )

            stdout = ""
            if isinstance(pipeline_result, dict):
                stdout = str(pipeline_result.get("stdout", ""))
            elif isinstance(pipeline_result, str):
                stdout = pipeline_result

            print("\nFinal Colab stdout:")
            print(stdout)


if __name__ == "__main__":
    asyncio.run(main())
