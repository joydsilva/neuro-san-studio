"""
Coded Tool wrapper for MCP Quote Generator
This tool interfaces with the MCP server to generate insurance quotes
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict

from neuro_san.interfaces.coded_tool import CodedTool

# Use NeuroSan's logging infrastructure
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SQUARE_FOOTAGE = 10000
DEFAULT_DEDUCTIBLE = 1000
DEFAULT_POLICY_TERM_DAYS = 365
RISK_TYPE = "General Business Risk"
LOCATION_REFERENCE = "LOC-001"


class McpQuoteGenerator(CodedTool):
    """
    CodedTool that generates insurance quotes using the MCP Quote Generation client.
    """

    def __init__(self):
        super().__init__()
        self.mcp_server_path = os.getenv(
            "MCP_SERVER_PATH", 
            "c:\\POC\\quote_mcp_server_simple\\index.js"
        )

    @staticmethod
    def _parse_amount(value: Any, default: int = 1000000) -> int:
        """Parse monetary amount from various input formats."""
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            return int(value.replace("$", "").replace(",", ""))
        return default

    async def async_invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> str:
        """Generate insurance quote using MCP server."""
        logger.info("MCP Quote Generator tool invoked!")
        
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
            
            # Validate required parameters
            required_params = ["business_type", "coverage_type", "coverage_limit", "location"]
            missing_params = [param for param in required_params if not args.get(param)]
            
            if missing_params:
                error_msg = f"❌ Error: Missing required parameters: {', '.join(missing_params)}"
                logger.error(error_msg)
                return error_msg

            # Create MCP client connection
            client = MultiServerMCPClient(
                {
                    "create-quote": {
                        "transport": "stdio",
                        "command": "node",
                        "args": [self.mcp_server_path],
                        "env": {}
                    }
                }
            )
            
            logger.info("Connecting to MCP server...")
            tools = await client.get_tools()
            
            if not tools:
                return "❌ Error: Quote generation service is not available. Please ensure the MCP server is running."

            # Find the quote creation tool
            quote_tool = next(
                (tool for tool in tools if "quote" in tool.name.lower() and "create" in tool.name.lower()),
                None
            )
            
            if not quote_tool:
                available_tools = ", ".join([t.name for t in tools])
                return f"❌ Error: Quote creation tool not found in MCP server. Available tools: {available_tools}"

            # Prepare quote request
            created_date = datetime.now().strftime("%Y-%m-%d")
            expiration_date = (datetime.now() + timedelta(days=DEFAULT_POLICY_TERM_DAYS)).strftime("%Y-%m-%d")
            
            quote_request = {
                "quote": {
                    "created_date": created_date,
                    "expiration_date": expiration_date,
                    "quote_status": "Active",
                    "locations": [
                        {
                            "address": args.get("location", "Unknown Location"),
                            "property_type": args.get("business_type", "Commercial Building"),
                            "square_footage": int(args.get("square_footage", DEFAULT_SQUARE_FOOTAGE))
                        }
                    ],
                    "lines": [
                        {
                            "line_type": args.get("coverage_type", "Property Insurance"),
                            "policy_limit": self._parse_amount(args.get("coverage_limit")),
                            "deductible": self._parse_amount(args.get("deductible"), DEFAULT_DEDUCTIBLE),
                            "risks": [
                                {
                                    "risk_type": RISK_TYPE,
                                    "location_reference": LOCATION_REFERENCE,
                                    "risk_description": f"{args.get('coverage_type', 'Property')} risk for {args.get('business_type', 'business')} operation",
                                    "coverages": [
                                        {
                                            "coverage_type": args.get("coverage_type", "Property"),
                                            "coverage_limit": self._parse_amount(args.get("coverage_limit"))
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                }
            }

            # Log the quote request
            logger.info("📤 Sending quote request to MCP server")
            logger.debug(f"Quote request data: {json.dumps(quote_request, indent=2)}")

            # Generate quote via MCP
            logger.info(f"Generating quote for {args.get('business_type')} - {args.get('coverage_type')}")
            quote_result = await quote_tool.ainvoke(quote_request)

            # Log the response
            logger.info("📥 Received response from MCP server")
            logger.debug(f"Quote result: {quote_result}")

            return str(quote_result)

        except ImportError as e:
            logger.error(f"Import error: {e}")
            return "❌ Error: MCP client not available. Please install langchain-mcp-adapters."
        except Exception as e:
            logger.error(f"Error generating quote: {str(e)}", exc_info=True)
            return f"❌ Error generating quote: {str(e)}\n\nPlease ensure the MCP quote server is configured correctly."