import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PyPSAResultAnalyzer:
    def __init__(self, network_path):
        """Initialize analyzer with PyPSA network"""
        self.n = pypsa.Network(network_path)
        self.snapshots = self.n.snapshots
        self.carriers = self.n.generators.carrier.unique()
        
        # Define color scheme for technologies
        self.tech_colors = {
            'onwind': '#3B6EA5',
            'offwind-ac': '#00A0B0', 
            'offwind-dc': '#004E5D',
            'solar': '#F49C3F',
            'CCGT': '#8B4513',
            'OCGT': '#CD853F',
            'coal': '#2F4F4F',
            'nuclear': '#B22222',
            'hydro': '#4682B4',
            'battery': '#9932CC',
            'H2': '#32CD32'
        }
    
    def create_capacity_pie_chart(self, save_path=None):
        """Create pie chart of optimal generation capacities by technology"""
        # Calculate total capacity by technology
        capacity_by_tech = self.n.generators.groupby('carrier')['p_nom_opt'].sum()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = [self.tech_colors.get(tech, '#808080') for tech in capacity_by_tech.index]
        
        wedges, texts, autotexts = ax.pie(
            capacity_by_tech.values, 
            labels=capacity_by_tech.index,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        
        ax.set_title('Optimal Generation Capacity Mix\n(Total: {:.1f} GW)'.format(
            capacity_by_tech.sum()/1000), fontsize=14, pad=20)
        
        # Make percentage text more readable
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return capacity_by_tech
    
    def create_generation_timeseries(self, save_path=None):
        """Create stacked area chart of generation dispatch over time"""
        # Get generation time series by carrier
        gen_by_carrier = pd.DataFrame()
        for carrier in self.carriers:
            carrier_gens = self.n.generators[self.n.generators.carrier == carrier].index
            if len(carrier_gens) > 0:
                gen_by_carrier[carrier] = self.n.generators_t.p[carrier_gens].sum(axis=1)
        
        # Convert to GW
        gen_by_carrier = gen_by_carrier / 1000
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Create stacked area plot
        ax.stackplot(gen_by_carrier.index, 
                    *[gen_by_carrier[col] for col in gen_by_carrier.columns],
                    labels=gen_by_carrier.columns,
                    colors=[self.tech_colors.get(col, '#808080') for col in gen_by_carrier.columns])
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Generation (GW)')
        ax.set_title('Electricity Generation Dispatch Over Time', fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return gen_by_carrier
    
    def create_regional_capacity_map(self, save_path=None):
        """Create bar chart showing capacity distribution by region and technology"""
        # Group generators by bus and carrier
        regional_capacity = self.n.generators.groupby(['bus', 'carrier'])['p_nom_opt'].sum().unstack(fill_value=0)
        
        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bottom = np.zeros(len(regional_capacity))
        for carrier in regional_capacity.columns:
            color = self.tech_colors.get(carrier, '#808080')
            ax.bar(regional_capacity.index, regional_capacity[carrier]/1000, 
                  bottom=bottom, label=carrier, color=color)
            bottom += regional_capacity[carrier]/1000
        
        ax.set_xlabel('Bus/Region')
        ax.set_ylabel('Installed Capacity (GW)')
        ax.set_title('Regional Distribution of Generation Capacity', fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return regional_capacity
    
    def create_price_analysis(self, save_path=None):
        """Create electricity price analysis plots"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Price time series
        prices = self.n.buses_t.marginal_price
        for bus in prices.columns:
            ax1.plot(prices.index, prices[bus], label=f'Bus {bus}', alpha=0.7)
        ax1.set_title('Electricity Prices Over Time')
        ax1.set_ylabel('Price (€/MWh)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Price distribution
        price_flat = prices.values.flatten()
        price_flat = price_flat[~np.isnan(price_flat)]
        ax2.hist(price_flat, bins=50, alpha=0.7, edgecolor='black')
        ax2.set_title('Price Distribution')
        ax2.set_xlabel('Price (€/MWh)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # 3. Average price by region
        avg_prices = prices.mean()
        ax3.bar(avg_prices.index, avg_prices.values, 
                color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'][:len(avg_prices)])
        ax3.set_title('Average Electricity Price by Region')
        ax3.set_xlabel('Bus/Region')
        ax3.set_ylabel('Average Price (€/MWh)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Price volatility
        price_std = prices.std()
        ax4.bar(price_std.index, price_std.values,
                color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'][:len(price_std)])
        ax4.set_title('Price Volatility by Region')
        ax4.set_xlabel('Bus/Region')
        ax4.set_ylabel('Price Std Dev (€/MWh)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return prices
    
    def create_transmission_analysis(self, save_path=None):
        """Analyze transmission line utilization and expansion"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Transmission capacity expansion
        if hasattr(self.n.lines, 's_nom_opt') and not self.n.lines.empty:
            capacity_expansion = (self.n.lines.s_nom_opt - self.n.lines.s_nom) / 1000  # Convert to GW
            ax1.bar(range(len(capacity_expansion)), capacity_expansion.values)
            ax1.set_title('Transmission Capacity Expansion by Line')
            ax1.set_xlabel('Line Index')
            ax1.set_ylabel('Capacity Addition (GW)')
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No transmission data available', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Transmission Capacity Expansion')
        
        # 2. Line utilization over time (if data exists)
        if hasattr(self.n, 'lines_t') and hasattr(self.n.lines_t, 'p0') and not self.n.lines_t.p0.empty:
            utilization = self.n.lines_t.p0.abs()
            avg_utilization = utilization.mean()
            ax2.bar(range(len(avg_utilization)), avg_utilization.values/1000)
            ax2.set_title('Average Line Utilization')
            ax2.set_xlabel('Line Index')
            ax2.set_ylabel('Average Power Flow (GW)')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No line flow data available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Line Utilization')
        
        # 3. Dynamic line rating impact (if available)
        if hasattr(self.n.lines_t, 's_max_pu') and not self.n.lines_t.s_max_pu.empty:
            dlr_variation = self.n.lines_t.s_max_pu.std()
            ax3.bar(range(len(dlr_variation)), dlr_variation.values)
            ax3.set_title('Dynamic Line Rating Variability')
            ax3.set_xlabel('Line Index')
            ax3.set_ylabel('Capacity Factor Std Dev')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No dynamic rating data available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Dynamic Line Rating Variability')
        
        # 4. Investment costs
        if hasattr(self.n.lines, 'capital_cost') and not self.n.lines.capital_cost.empty:
            ax4.bar(range(len(self.n.lines.capital_cost)), self.n.lines.capital_cost.values)
            ax4.set_title('Transmission Investment Costs')
            ax4.set_xlabel('Line Index')
            ax4.set_ylabel('Cost (€/MW/year)')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No cost data available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Investment Costs')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_environmental_analysis(self, save_path=None):
        """Analyze environmental performance metrics"""
        # Define emission factors (kg CO2/MWh)
        emission_factors = {
            'coal': 820,
            'CCGT': 350,
            'OCGT': 400,
            'nuclear': 0,
            'onwind': 0,
            'offwind-ac': 0,
            'offwind-dc': 0,
            'solar': 0,
            'hydro': 0
        }
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Generation mix pie chart
        total_gen = pd.Series(dtype=float)
        for carrier in self.carriers:
            carrier_gens = self.n.generators[self.n.generators.carrier == carrier].index
            if len(carrier_gens) > 0:
                total_gen[carrier] = self.n.generators_t.p[carrier_gens].sum().sum()
        
        colors = [self.tech_colors.get(tech, '#808080') for tech in total_gen.index]
        ax1.pie(total_gen.values, labels=total_gen.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        ax1.set_title('Generation Mix by Energy')
        
        # 2. Renewable share over time
        renewable_techs = ['onwind', 'offwind-ac', 'offwind-dc', 'solar', 'hydro']
        renewable_gen = pd.Series(0, index=self.snapshots)
        total_gen_hourly = pd.Series(0, index=self.snapshots)
        
        for carrier in self.carriers:
            carrier_gens = self.n.generators[self.n.generators.carrier == carrier].index
            if len(carrier_gens) > 0:
                gen = self.n.generators_t.p[carrier_gens].sum(axis=1)
                total_gen_hourly += gen
                if carrier in renewable_techs:
                    renewable_gen += gen
        
        renewable_share = (renewable_gen / total_gen_hourly * 100).fillna(0)
        ax2.plot(renewable_share.index, renewable_share.values)
        ax2.set_title('Renewable Share Over Time')
        ax2.set_ylabel('Renewable Share (%)')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # 3. CO2 emissions by technology
        emissions_by_tech = {}
        for carrier in self.carriers:
            if carrier in emission_factors:
                carrier_gens = self.n.generators[self.n.generators.carrier == carrier].index
                if len(carrier_gens) > 0:
                    gen = self.n.generators_t.p[carrier_gens].sum().sum() / 1000  # Convert to GWh
                    emissions_by_tech[carrier] = gen * emission_factors[carrier] / 1000  # Convert to kt CO2
        
        if emissions_by_tech:
            ax3.bar(emissions_by_tech.keys(), emissions_by_tech.values(),
                   color=[self.tech_colors.get(tech, '#808080') for tech in emissions_by_tech.keys()])
            ax3.set_title('CO₂ Emissions by Technology')
            ax3.set_ylabel('Emissions (kt CO₂)')
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
            ax3.grid(True, alpha=0.3)
        
        # 4. Capacity factors
        capacity_factors = {}
        for carrier in self.carriers:
            carrier_gens = self.n.generators[self.n.generators.carrier == carrier].index
            if len(carrier_gens) > 0:
                gen_data = self.n.generators_t.p[carrier_gens]
                capacity_data = self.n.generators.loc[carrier_gens, 'p_nom_opt']
                if capacity_data.sum() > 0:
                    cf = gen_data.sum().sum() / (capacity_data.sum() * len(self.snapshots)) * 100
                    capacity_factors[carrier] = cf
        
        if capacity_factors:
            ax4.bar(capacity_factors.keys(), capacity_factors.values(),
                   color=[self.tech_colors.get(tech, '#808080') for tech in capacity_factors.keys()])
            ax4.set_title('Average Capacity Factors')
            ax4.set_ylabel('Capacity Factor (%)')
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'total_generation': total_gen,
            'renewable_share': renewable_share.mean(),
            'emissions': sum(emissions_by_tech.values()) if emissions_by_tech else 0,
            'capacity_factors': capacity_factors
        }
    
    def create_storage_analysis(self, save_path=None):
        """Analyze storage operation and sizing"""
        if self.n.storage_units.empty:
            print("No storage units found in the network")
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Storage capacity by type
        storage_capacity = self.n.storage_units.groupby('carrier')['p_nom_opt'].sum()
        colors = [self.tech_colors.get(tech, '#808080') for tech in storage_capacity.index]
        ax1.bar(storage_capacity.index, storage_capacity.values/1000, color=colors)
        ax1.set_title('Storage Capacity by Technology')
        ax1.set_ylabel('Capacity (GW)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Storage operation over time
        if hasattr(self.n.storage_units_t, 'p') and not self.n.storage_units_t.p.empty:
            storage_op = self.n.storage_units_t.p.sum(axis=1) / 1000
            ax2.plot(storage_op.index, storage_op.values)
            ax2.set_title('Total Storage Operation')
            ax2.set_ylabel('Power (GW, +charging/-discharging)')
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax2.grid(True, alpha=0.3)
        
        # 3. Storage state of charge
        if hasattr(self.n.storage_units_t, 'state_of_charge') and not self.n.storage_units_t.state_of_charge.empty:
            soc = self.n.storage_units_t.state_of_charge.sum(axis=1) / 1000
            ax3.plot(soc.index, soc.values)
            ax3.set_title('Total Storage State of Charge')
            ax3.set_ylabel('Energy (GWh)')
            ax3.grid(True, alpha=0.3)
        
        # 4. Storage utilization
        if hasattr(self.n.storage_units_t, 'p') and not self.n.storage_units_t.p.empty:
            utilization = {}
            for storage in self.n.storage_units.index:
                if storage in self.n.storage_units_t.p.columns:
                    capacity = self.n.storage_units.loc[storage, 'p_nom_opt']
                    if capacity > 0:
                        avg_power = abs(self.n.storage_units_t.p[storage]).mean()
                        utilization[storage] = avg_power / capacity * 100
            
            if utilization:
                ax4.bar(range(len(utilization)), list(utilization.values()))
                ax4.set_title('Storage Utilization Rate')
                ax4.set_xlabel('Storage Unit')
                ax4.set_ylabel('Avg Utilization (%)')
                ax4.set_xticks(range(len(utilization)))
                ax4.set_xticklabels([f'S{i}' for i in range(len(utilization))], rotation=45)
                ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return storage_capacity
    
    def create_cost_breakdown(self, save_path=None):
        """Create comprehensive cost analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Total system cost
        total_cost = self.n.objective / 1e9  # Convert to billion €
        ax1.text(0.5, 0.5, f'Total Annual\nSystem Cost\n\n€{total_cost:.2f} billion', 
                ha='center', va='center', fontsize=16, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        ax1.set_title('Annual System Cost')
        
        # 2. Capital costs by technology
        cap_costs = {}
        for carrier in self.carriers:
            carrier_gens = self.n.generators[self.n.generators.carrier == carrier].index
            if len(carrier_gens) > 0 and 'capital_cost' in self.n.generators.columns:
                total_cap_cost = (self.n.generators.loc[carrier_gens, 'p_nom_opt'] * 
                                self.n.generators.loc[carrier_gens, 'capital_cost']).sum()
                if total_cap_cost > 0:
                    cap_costs[carrier] = total_cap_cost / 1e6  # Convert to million €
        
        if cap_costs:
            colors = [self.tech_colors.get(tech, '#808080') for tech in cap_costs.keys()]
            ax2.bar(cap_costs.keys(), cap_costs.values(), color=colors)
            ax2.set_title('Annual Capital Costs by Technology')
            ax2.set_ylabel('Cost (M€/year)')
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            ax2.grid(True, alpha=0.3)
        
        # 3. Marginal costs by technology
        marg_costs = {}
        for carrier in self.carriers:
            carrier_gens = self.n.generators[self.n.generators.carrier == carrier].index
            if len(carrier_gens) > 0 and 'marginal_cost' in self.n.generators.columns:
                avg_marg_cost = self.n.generators.loc[carrier_gens, 'marginal_cost'].mean()
                if pd.notna(avg_marg_cost):
                    marg_costs[carrier] = avg_marg_cost
        
        if marg_costs:
            colors = [self.tech_colors.get(tech, '#808080') for tech in marg_costs.keys()]
            ax3.bar(marg_costs.keys(), marg_costs.values(), color=colors)
            ax3.set_title('Average Marginal Costs')
            ax3.set_ylabel('Cost (€/MWh)')
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
            ax3.grid(True, alpha=0.3)
        
        # 4. Cost per unit of capacity
        cost_per_mw = {}
        for carrier in self.carriers:
            if carrier in cap_costs:
                carrier_gens = self.n.generators[self.n.generators.carrier == carrier].index
                total_capacity = self.n.generators.loc[carrier_gens, 'p_nom_opt'].sum()
                if total_capacity > 0:
                    cost_per_mw[carrier] = cap_costs[carrier] * 1000 / total_capacity  # €/MW/year
        
        if cost_per_mw:
            colors = [self.tech_colors.get(tech, '#808080') for tech in cost_per_mw.keys()]
            ax4.bar(cost_per_mw.keys(), cost_per_mw.values(), color=colors)
            ax4.set_title('Capital Cost per Unit Capacity')
            ax4.set_ylabel('Cost (k€/MW/year)')
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'total_cost': total_cost,
            'capital_costs': cap_costs,
            'marginal_costs': marg_costs
        }
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("="*80)
        print("PYPSA-EUR RESULTS SUMMARY REPORT")
        print("="*80)
        
        # Basic network info
        print(f"\n📊 NETWORK OVERVIEW")
        print(f"   • Number of buses: {len(self.n.buses)}")
        print(f"   • Number of generators: {len(self.n.generators)}")
        print(f"   • Number of lines: {len(self.n.lines)}")
        print(f"   • Time period: {self.snapshots[0]} to {self.snapshots[-1]}")
        print(f"   • Number of snapshots: {len(self.snapshots)}")
        
        # Capacity mix
        capacity_by_tech = self.n.generators.groupby('carrier')['p_nom_opt'].sum() / 1000
        print(f"\n⚡ GENERATION CAPACITY MIX")
        for tech, capacity in capacity_by_tech.items():
            print(f"   • {tech:12}: {capacity:8.1f} GW")
        print(f"   • {'TOTAL':12}: {capacity_by_tech.sum():8.1f} GW")
        
        # Economic summary
        total_cost = self.n.objective / 1e9
        print(f"\n💰 ECONOMIC SUMMARY")
        print(f"   • Total annual system cost: €{total_cost:.2f} billion")
        
        # Environmental metrics
        renewable_techs = ['onwind', 'offwind-ac', 'offwind-dc', 'solar', 'hydro']
        renewable_capacity = capacity_by_tech[capacity_by_tech.index.isin(renewable_techs)].sum()
        renewable_share = renewable_capacity / capacity_by_tech.sum() * 100
        
        print(f"\n🌱 ENVIRONMENTAL METRICS")
        print(f"   • Renewable capacity share: {renewable_share:.1f}%")
        print(f"   • Renewable capacity: {renewable_capacity:.1f} GW")
        
        print("\n" + "="*80)

# Main execution function
def main():
    """Main function to run all analyses"""
    # Load network
    network_path = "pypsa-eur/results/test-elec/networks/base_s_6_elec_.nc"
    
    try:
        analyzer = PyPSAResultAnalyzer(network_path)
        print("Successfully loaded PyPSA network!")
        
        # Create output directory
        import os
        os.makedirs("analysis_results", exist_ok=True)
        
        # Generate all visualizations
        print("\n🎨 Creating visualizations...")
        
        print("1. Capacity pie chart...")
        analyzer.create_capacity_pie_chart("analysis_results/capacity_pie.png")
        
        print("2. Generation time series...")
        analyzer.create_generation_timeseries("analysis_results/generation_timeseries.png")
        
        print("3. Regional capacity analysis...")
        analyzer.create_regional_capacity_map("analysis_results/regional_capacity.png")
        
        print("4. Price analysis...")
        analyzer.create_price_analysis("analysis_results/price_analysis.png")
        
        print("5. Transmission analysis...")
        analyzer.create_transmission_analysis("analysis_results/transmission_analysis.png")
        
        print("6. Environmental analysis...")
        analyzer.create_environmental_analysis("analysis_results/environmental_analysis.png")
        
        print("7. Storage analysis...")
        analyzer.create_storage_analysis("analysis_results/storage_analysis.png")
        
        print("8. Cost breakdown...")
        analyzer.create_cost_breakdown("analysis_results/cost_breakdown.png")
        
        # Generate summary report
        print("\n📋 Generating summary report...")
        analyzer.generate_summary_report()
        
        print("\n✅ Analysis complete! Check 'analysis_results/' folder for visualizations.")
        
    except FileNotFoundError:
        print(f"❌ Network file not found: {network_path}")
        print("Please make sure you have run the PyPSA-EUR workflow first.")
    except Exception as e:
        print(f"❌ Error analyzing network: {str(e)}")

if __name__ == "__main__":
    main()