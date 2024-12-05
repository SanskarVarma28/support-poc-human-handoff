import re

import numpy as np
import openai
from langchain_core.tools import tool
import requests

faq_text = """
# SuperAGI

SuperAGI builds AI-native platform for Sales, Support and Marketing. We are a company based out of Palo Alto, California. We gained popularity with our open-source framework for building AI agents (https://github.com/TransformerOptimus/SuperAGI) which has over 15K stars on Github.

## SuperAGI Platform

SuperAGI offers three products – SuperSales, SuperSupport, and SuperMarketer. SuperSales is an all-in-one sales platform which helps you manage your inbound and outbound sales. SuperMarketer is an AI-native engagement marketing platform. You can run AI-personalized campaigns across multiple channels. SuperSupport is an AI-agent based support software where AI-agents resolver customer issues instantly by taking actions like a seasoned support staff and it is available to serve each and every customer 24x7. All the three platforms work together to provide a unified view of your customers and let’s you leverage AI agents to automate every workflow which are currently done by humans. Our AI Agents work along with your human employees to scale your organization without growing your headcount.

## SuperAGI related FAQs-

### 1. How do i Signup for Supersales?

You can signup by clicking on the “Start for Free” button on our landing page (superagi.com). 

### 2. Is there a free trial for SuperSales?

Yes, we give you a 14 days of free trial post which you automatically get transferred to a free tier.

### 3. Does SuperSales have seat based pricing?

No, there is no limit on the no. of seats (accounts) connected to a SuperSales workspace.

### 4. What are the various integrations available with SuperSales?

We have multiple integrations. Some of the popular ones are:

1. SalesForce and Hubspot: You can import your contacts and companies data from your existing CRM. This also ensures that the AI SDR does not reach out to the contacts who are already in your pipeline.
2. Gmail and Outlook: You can connect your multiple mailboxes for a single unified view of all your conversations with the customers. If you’re using any other email provider, you can use SMTP to connect those inboxes.
3. Calendar: You can sync your calendar with SuperSales which allows you to create meeting agents for booking meetings with customers. These agents can be shared via their unique links or by embedding on your websites.

### 5. How does SuperAGI’s platform deliver Business Super Intelligence?

Our products are built on a common platform architecture that combines a System of Record (SoR) for storing all the critical business data, and System of Action (SoA) via SuperAGI’s agent framework to form a System of Intelligence (SoI). SoR, SoA & SoI constitute what we call Business Super Intelligence.

### 6. What are the different pricing plans available for SuperSales? What are the differences in terms of features.

- Free Tier: Basic features like Lead Database Access, Basic Data Enrichment, Market Research, Meeting Scheduling, CRM Functionality, Mobile and Chrome Extensions
- Lite Plan ($75/month): Adds AI SDR
- Starter Plan ($150/month): Includes inbox rotation and analytics
- Growth Plan ($350/month): Adds phone dialer and LinkedIn automation
- Business Plan ($500/month): Includes sequences and salesforce integration
- Enterprise: Custom pricing, tailored as per the specific needs

For more details, you can visit https://superagi.com/pricing/

### **7. How does SuperSupport's omnichannel support functionality work?**

Our SuperSupport integrates communication channels like email, chat, calls, and mobile SDKs into a unified inbox for seamless interactions.

- Real-time synchronization ensures consistent customer context across channels.
- Intelligent routing assigns tickets efficiently, while AI automates routine queries.
- Mobile SDKs allow in-app chat and notifications, enhancing customer convenience.

### 8. What kind of lead scoring does SuperSales provide?

Our SuperSales incorporates AI-driven lead scoring, which automatically assigns scores to leads based on their behavior, demographics, and interactions with the brand. This includes factors like website visits, email opens, social media engagement, and previous purchase history. By analyzing these activities, the platform helps sales teams identify the most promising leads, allowing them to prioritize their outreach and tailor their sales pitches accordingly for more effective conversions.

### 9. How does SuperSales integrate with LinkedIn for sales outreach?

Our SuperSales integrates seamlessly with LinkedIn, enabling users to automate routine tasks such as sending connection requests, personalized InMails, and follow-ups. It works in conjunction with LinkedIn Sales Navigator, allowing users to identify and target qualified leads based on filters like industry, geography, and seniority. The platform also tracks LinkedIn engagement, such as responses and profile visits, and logs this data into the CRM for holistic tracking. This integration helps sales teams scale outreach while maintaining personalization and staying compliant with LinkedIn’s terms of use.

### **10. Can you explain the AI-suggested actions and sales co-pilot features in SuperSales?**

Our AI-suggested actions feature analyzes historical interactions, call transcripts, and deal stages to recommend precise next steps, such as sending follow-up emails, scheduling demos, or offering discounts. It uses predictive analytics to identify opportunities or risks, ensuring deals stay on track. Our sales co-pilot acts as a real-time assistant during live calls or meetings. It provides contextual guidance, such as key talking points, handling objections, or suggesting upsell opportunities. These features enhance productivity and ensure sales reps make data-driven decisions.

### 11. Can I book a demo?

Yes, you can go to our homepage (superagi.com) and click on Get a Demo button.

### 12. What are the key features of the open-source SuperAGI framework?

Here are the key innovations of the SuperAGI framework:

1. Provisioning and Deployment:
SuperAGI allows developers to provision, spawn, and deploy autonomous AI agents with ease, streamlining the development process.
2. Extensibility:
Developers can extend agent capabilities using a variety of tools, enhancing functionality and adaptability for specific tasks.
3. Concurrent Agent Management:
The framework supports running multiple agents simultaneously, maximizing efficiency and resource utilization.
4. User-Friendly Interface:
A graphical user interface (GUI) simplifies interaction with agents, allowing for easier management and monitoring.
5. Action Console:
This feature provides real-time insights into agent actions, facilitating better control and understanding of agent behavior.
6. Advanced Capabilities:
SuperAGI includes features such as multi-modal agents, performance telemetry, memory storage, and looping detection heuristics, which contribute to its versatility.

# SuperAGIs Products:

1. SuperSales: AI-native all-in-one sales platform with built-in AI SDR and multiple other agents.
2. SuperSupport: AI Native Customer Support App for autonomous instant issue resolution.
3. SuperMartketer: AI Native retention marketing and customer engagement
4. SuperCoder: End to end autonomous development platform

## SuperSales

### SuperSales: Overview

SuperSales combines has five major modules:

1. Engagement:
    1. AI Outbound & Inbound SDR: For autonomous pipeline generation
    2. Sequences: Human controlled outbound sales
    3. Dialer: For parallel and power dialing with AI personalized sales transcripts
    4. Meetings: For booking inbound meetings and distributing them via round robin to AEs
    5. Conversations: For unified view of all the communication to every lead via any channel
    6. Email warmup infra: A toolset to enhance your email deliverability
2. CRM
    1. Account Management: Manage all your contacts
    2. Pipeline Management: Manage deals across pipeline stages
    3. Lead Management: Manage leads and use them for prospecting
    4. Mobile App: Access your CRM on the go – a life saver on field sales team
    5. Browser Extension: Add leads to your CRM directly from your browser
3. Data
    1. Contact and Account Search: A large B2B database with 300M+ contacts from 75M+ companies with 97% email accuracy
    2. Buyer Intent & Signals: Find signals like website visitors, job changes, and hiring announcements to help you with (ABM) Account Based Marketing
    3. Data Enrichment: Collate multiple sources for most accurate data of your contacts.
4. AI and Automation:
    1. Agent Builder: Build custom agents to automate your workflows
    2. Next Best Action: With Business SuperIntelligence, you get to know a prioritized set of actions for every contact and account
5. Analytics:
    1. Sales Forecasting: Get an accurate estimate of your future sales numbers
    2. AI Insights & Copilot: Our sales copilot can answer your custom queries and provide you with data backed insights
    3. Revenue Analytics: Get a deeper understanding of the sales activity performed by your sales team in form of pre-generated reports

### SuperSales: Comparisons with other players

SuperSales simplifies your sales stack by replacing multiple products across ten sales categories:

1. Legacy CRMs: Hubspot, SalesForce, Highlevel, Pipedrive, Attio are SaaS era legacy CRMs. SuperSales is the first AI-native CRM with a built-in agent curated data source. With SuperSales you can generate more pipeline, get better conversions with personalized outreaches based on deep research, and close more deals – all on autopilot!
2. Contact Data: Players like Apollo, Zoominfo, Uplead, Rocketreach, etc. are mostly data vendors. They do very little to aid you in downstream workflows. SuperSales has agent curated data which is more accurate than legacy rivals. You don’t need to have a separate tool for finding new leads and accounts.
3. Enrichment Tools: SuperSales subsumes the functionality offered by enrichment tools like Lusha, [Seamless.ai](http://seamless.ai/) and Clay. You can bring in your own list and we will populate the relevant for you by using ai agents.
4. Sales Engagement: Tools like outreach, [close.io](http://close.io/), and salesloft let you automate your engagement workflows. SuperSales allows more intelligent automation which is based on more accurate present in the SuperSales CRM.
5. Email Deliverability: No need to purchase another tool for just email warmup like Lemlist, Instantly, or warmly. SuperSales allows you to improve email and domain health so that your mails land in the inbox instead of spam.
6. Caller: Calling is an integral part of B2B sales which is best done by humans and supersales lets your sales reps focus on selling, by handling the supporting functions like dialing, note taking, and call analytics in the background.
7. AI SDRs: There are multiple point solutions which offer AI agents to do personalized mass prospecting. However, SuperSales is the only platform which has complete picture of your pipeline and all other sales activities. That is why our AI SDRs (outbound and inbound) outperform players like 11x, Artisan, and [Relevance.ai](http://relevance.ai/) by miles.
8. Meetings: We offer a meeting agent which can allow you to book meetings via a common meeting link. We support multiple kinds of meetings like fixed host or round robin (where inbound leads are distributed among the sales team). You no longer need a separate tool like Calendly, Chillipiper, or [cal.com](http://cal.com/)
9. Research: Our cutting edge AI agents are far superior and more relevant for sales, than the agents offered by generic research tools like Aomni, [Regie.ai](http://regie.ai/), and Jasper.
10. LinkedIn Automation: You no longer need another tool like Octopus, [Expandi.io](http://expandi.io/), or Phantombuster to automate your linkedin outreaches. SuperSales has these capabilities built in.

SuperSales is the only sales tool which you will ever need.

### SuperSales: Pricing

For SuperSales, We have four paid plans along with a generous free tier.

In the free tier you’ll get:

1. Access to Lead Database
2. Data Enrichment
3. Market Research
4. Meeting Scheduling
5. CRM
6. Chrome Plugin
7. Mobile App
8. Customer Support

In the Lite plan (priced at $75 monthly) you’ll get everything in Free plan plus AI SDR.

In the Starter Plan (priced at $150 monthly) you’ll get:

1. Everything in Lite plan; plus
2. Inbox Rotation (for more email reach outs without risking domain reputations which leads to your emails going into spam folder)
3. Analytics

In the Growth Plan (priced at $350 monthly) you’ll get:

1. Everything in the Starter Plan; plus
2. Phone Dialer
3. LinkedIn Outreach Automation
4. Hubspot Integration
The growth plan is also the most popular plan.

In the Business Plan (priced at $500 monthly) you’ll get:

1. Sequences
2. Salesforce Integration

For Enterprises, we also offer custom pricing plans which you can discuss with our sales team.

## SuperMarketer

### SuperMarketer: Overview

1. Customer engagement:
    1. Automation, (Green Pill)
    2. Campaigns,
    3. Forms
    4. Workflow automation- generative customer journeys
2. Channels and Languages
    1. Omnichannel - email, SMS, web push
    2. Mobile SDK with Push Notifications
    3. Multi-Lingual Generation
3. Customer understanding:
    1. Dynamic Customer Segmentation
    2. A/B testing predict winning copies
    3. Web SDK to track custom events
4. Content creation :
    1. Landing page,
    2. podcasts,
    3. blog post, case studies
5. Analytics
    1. Session Analytics
    2. Scheduled Analytics and Reporting
    3. Engagement Analytics

## SuperSupport

### SuperSupport: Overview

1. Conversations
    1. Inbox Views
    2. Human hand -off
    3. Tagging
    4. Multiple team inboxes
2. Tickets
    1. Workload Management (Routing)
    2. Tracking
    3. Back office collaboration
3. Helpdesk
    1. Knowledge Base
    2. Knowledge updating with Human feedback
    3. Help Centre
4. Omni - Channel
    1. Emails
    2. Web Chat
    3. Calls
    4. Mobile SDKs
5. Analytics
    1. CSAT
    2. SLAs and custom reports
    3. Agent Performance
"""

docs = [{"page_content": txt} for txt in re.split(r"(?=\n##)", faq_text)]


class VectorStoreRetriever:
    def __init__(self, docs: list, vectors: list, oai_client):
        self._arr = np.array(vectors)
        self._docs = docs
        self._client = oai_client

    @classmethod
    def from_docs(cls, docs, oai_client):
        embeddings = oai_client.embeddings.create(
            model="text-embedding-3-small", input=[doc["page_content"] for doc in docs]
        )
        vectors = [emb.embedding for emb in embeddings.data]
        return cls(docs, vectors, oai_client)

    def query(self, query: str, k: int = 5) -> list[dict]:
        embed = self._client.embeddings.create(
            model="text-embedding-3-small", input=[query]
        )
        # "@" is just a matrix multiplication in python
        scores = np.array(embed.data[0].embedding) @ self._arr.T
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx_sorted = top_k_idx[np.argsort(-scores[top_k_idx])]
        return [
            {**self._docs[idx], "similarity": scores[idx]} for idx in top_k_idx_sorted
        ]


retriever = VectorStoreRetriever.from_docs(docs, openai.Client())


@tool
def lookup_knowledge_base(query: str) -> str:
    """Consult the knowledge base to answer customer queries."""
    docs = retriever.query(query, k=2)
    return "\n\n".join([doc["page_content"] for doc in docs])