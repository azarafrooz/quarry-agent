import asyncio
from crawl4ai import AsyncWebCrawler
import re


class GetTableDocumentation:
    def __init__(self):
        self.pattern = r"# Table(.*?)## Export"

    async def crawl(self, table: str) -> str:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(
                url=f"https://hub.steampipe.io/plugins/turbot/{table.split('_')[0]}/tables/{table}",
            )
            return result.markdown

    def parse(self, content):
        match = re.search(self.pattern, content, re.DOTALL)
        if match:
            content = match.group(1).strip()  # Extract and clean up the content
            return "Table " + content
        else:
            return ""

    def __call__(self, table_name):
        content = asyncio.run(self.crawl(table_name))
        content = self.parse(content)
        return content
