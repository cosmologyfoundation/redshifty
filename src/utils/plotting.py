"""
Plotting Utilities for DESI Spectra
====================================
Visualization tools for spectra, redshift distributions, and reconstructions.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Union
import torch


def plot_spectrum(
    wavelength: np.ndarray,
    flux: np.ndarray,
    ivar: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    z: Optional[float] = None,
    ax=None,
    title: Optional[str] = None,
    show_errors: bool = True,
    color: str = "black",
    alpha: float = 1.0,
    label: Optional[str] = None,
):
    """Plot a single spectrum with optional error bars and masking.
    
    Args:
        wavelength: Wavelength array in Angstroms
        flux: Flux array
        ivar: Inverse variance array (optional)
        mask: Boolean mask array (optional, True = bad pixel)
        z: Redshift value for title (optional)
        ax: Matplotlib axis (optional)
        title: Plot title (optional)
        show_errors: Whether to show 1-sigma error regions
        color: Line color
        alpha: Line alpha
        label: Legend label
        
    Returns:
        matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    
    # Plot unmasked regions
    if mask is not None:
        good = ~mask
        ax.plot(wavelength[good], flux[good], color=color, alpha=alpha, linewidth=0.8, label=label)
        if good.any() and (~good).any():
            ax.plot(wavelength[~good], flux[~good], color="red", alpha=0.3, linewidth=0.5, label="masked")
    else:
        ax.plot(wavelength, flux, color=color, alpha=alpha, linewidth=0.8, label=label)
    
    # Show error region
    if show_errors and ivar is not None:
        sigma = 1.0 / np.sqrt(ivar + 1e-20)
        good = ~mask if mask is not None else np.ones(len(wavelength), dtype=bool)
        ax.fill_between(
            wavelength[good],
            flux[good] - sigma[good],
            flux[good] + sigma[good],
            alpha=0.2,
            color=color,
        )
    
    ax.set_xlabel(r"Wavelength [\AA]", fontsize=12)
    ax.set_ylabel(r"Flux [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$]", fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14)
    elif z is not None:
        ax.set_title(f"DESI Spectrum (z = {z:.4f})", fontsize=14)
    
    ax.set_xlim(wavelength.min(), wavelength.max())
    
    # Auto-scale y-axis to exclude extreme outliers
    if mask is not None:
        y_good = flux[~mask]
    else:
        y_good = flux
    
    if len(y_good) > 0:
        y_med = np.median(y_good)
        y_std = np.std(y_good)
        ax.set_ylim(y_med - 5*y_std, y_med + 5*y_std)
    
    if label:
        ax.legend(loc="upper right")
    
    return ax


def plot_spectrum_grid(
    spectra: List[Dict[str, np.ndarray]],
    ncols: int = 3,
    figsize_per_spectrum: tuple = (4, 3),
    save_path: Optional[Union[str, Path]] = None,
):
    """Plot a grid of spectra.
    
    Args:
        spectra: List of dicts with keys: wavelength, flux, ivar, mask, z
        ncols: Number of columns
        figsize_per_spectrum: (width, height) per subplot in inches
        save_path: Optional path to save figure
        
    Returns:
        matplotlib figure
    """
    n_spectra = len(spectra)
    nrows = (n_spectra + ncols - 1) // ncols
    
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_spectrum[0] * ncols, figsize_per_spectrum[1] * nrows),
        squeeze=False,
    )
    
    for i, spec in enumerate(spectra):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col]
        
        plot_spectrum(
            wavelength=spec["wavelength"],
            flux=spec["flux"],
            ivar=spec.get("ivar"),
            mask=spec.get("mask"),
            z=spec.get("z"),
            ax=ax,
            title=f"z = {spec.get('z', 0):.4f}" if "z" in spec else None,
        )
    
    # Hide unused subplots
    for i in range(n_spectra, nrows * ncols):
        row = i // ncols
        col = i % ncols
        axes[row, col].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_redshift_distribution(
    redshifts: np.ndarray,
    bins: int = 30,
    ax=None,
    title: str = "Redshift Distribution",
    save_path: Optional[Union[str, Path]] = None,
):
    """Plot histogram of redshift distribution.
    
    Args:
        redshifts: Array of redshift values
        bins: Number of histogram bins
        ax: Matplotlib axis (optional)
        title: Plot title
        save_path: Optional path to save figure
        
    Returns:
        matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.hist(redshifts, bins=bins, color="steelblue", edgecolor="black", alpha=0.7)
    ax.set_xlabel("Redshift z", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Add statistics text
    z_mean = np.mean(redshifts)
    z_std = np.std(redshifts)
    ax.axvline(z_mean, color="red", linestyle="--", linewidth=2, label=f"mean = {z_mean:.3f}")
    ax.text(
        0.95, 0.95,
        f"N = {len(redshifts)}\nmean = {z_mean:.3f}\nstd = {z_std:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax.legend(loc="upper left")
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    
    return ax


def plot_reconstruction_comparison(
    wavelength: np.ndarray,
    original: np.ndarray,
    reconstructed: np.ndarray,
    mask: Optional[np.ndarray] = None,
    ivar: Optional[np.ndarray] = None,
    z: Optional[float] = None,
    save_path: Optional[Union[str, Path]] = None,
):
    """Plot original vs reconstructed spectrum.
    
    Args:
        wavelength: Wavelength array
        original: Original flux
        reconstructed: Reconstructed flux
        mask: Boolean mask (optional)
        ivar: Inverse variance (optional)
        z: Redshift (optional)
        save_path: Optional path to save figure
        
    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 1]})
    
    # Original
    plot_spectrum(wavelength, original, ivar=ivar, mask=mask, z=z,
                  ax=axes[0], color="black", label="Original")
    axes[0].set_title("Original Spectrum", fontsize=14)
    
    # Reconstructed
    plot_spectrum(wavelength, reconstructed, ivar=ivar, mask=mask, z=z,
                  ax=axes[1], color="blue", label="Reconstructed")
    axes[1].set_title("Reconstructed Spectrum", fontsize=14)
    
    # Residual
    residual = original - reconstructed
    if mask is not None:
        good = ~mask
    else:
        good = np.ones(len(wavelength), dtype=bool)
    
    axes[2].plot(wavelength[good], residual[good], color="green", linewidth=0.8, alpha=0.7)
    axes[2].axhline(0, color="black", linestyle="--", linewidth=1)
    
    if ivar is not None:
        sigma = 1.0 / np.sqrt(ivar[good] + 1e-20)
        axes[2].fill_between(
            wavelength[good],
            -sigma,
            sigma,
            alpha=0.2,
            color="gray",
            label="1σ noise",
        )
    
    axes[2].set_xlabel(r"Wavelength [\AA]", fontsize=12)
    axes[2].set_ylabel("Residual", fontsize=12)
    axes[2].set_title("Original - Reconstructed", fontsize=14)
    axes[2].legend(loc="upper right")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    train_z_losses: Optional[List[float]] = None,
    val_z_losses: Optional[List[float]] = None,
    save_path: Optional[Union[str, Path]] = None,
):
    """Plot training and validation loss curves.
    
    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch (optional)
        train_z_losses: Training redshift losses (optional)
        val_z_losses: Validation redshift losses (optional)
        save_path: Optional path to save figure
        
    Returns:
        matplotlib figure
    """
    n_plots = 1 + (train_z_losses is not None)
    fig, axes = plt.subplots(1, n_plots, figsize=(7*n_plots, 5), squeeze=False)
    
    # Total loss
    ax = axes[0, 0]
    ax.plot(train_losses, label="Train", color="steelblue", linewidth=2)
    if val_losses:
        ax.plot(val_losses, label="Val", color="orange", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Total Loss", fontsize=12)
    ax.set_title("Training Loss", fontsize=14)
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    
    # Redshift loss
    if train_z_losses is not None:
        ax = axes[0, 1]
        ax.plot(train_z_losses, label="Train", color="steelblue", linewidth=2)
        if val_z_losses:
            ax.plot(val_z_losses, label="Val", color="orange", linewidth=2)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Redshift MSE Loss", fontsize=12)
        ax.set_title("Redshift Loss", fontsize=14)
        ax.legend()
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    
    return fig
